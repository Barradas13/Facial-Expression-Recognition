import shutil
import cv2
import mediapipe as mp
import math
import tqdm
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any, Iterator
from contextlib import contextmanager
import logging
import pandas as pd
import re 

# Config file
import Config as cnf

# !!! The code runs on older version of mediapipe, use python -m pip install mediapipe==0.10.21 !!!

class Image:
    @staticmethod
    def read(path: str): # Read an image from path
        try:
            data = np.fromfile(path, np.uint8)
        except OSError:
            print("File reading exception")
            return None

        img = cv2.imdecode(data, cv2.IMREAD_COLOR) # Read BGR

        if img is None or img.size == 0:
            print("Image file corrupted")
            return None

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Switch to RGB

    @staticmethod
    def listPaths(dir: str) -> List[Path]: # List of all image paths at directory
        root = Path(dir)

        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root.resolve()}")

        return sorted(
            p for p in root.rglob("*")
            if p.is_file()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
            and p.suffix.lower() in cnf.CONFIG["extensions"]
        )

    @staticmethod
    def readAll(dir: str, extensions): # Readout all images at directory
        for path in Image.listPaths(dir):
            img = Image.read(str(path))
            if img is None:
                print(f"Unreadable: {path}")
            yield path, img

    @staticmethod
    def validate(img: np.ndarray):
        issues: List[str] = []
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        height, width = gray.shape[:2]

        if blur_score < cnf.CONFIG["blur_threshold"]:
            issues.append(f"blurry: laplacian_var={blur_score:.1f} < {cnf.CONFIG['blur_threshold']:g}")
        if brightness < cnf.CONFIG["brightness_low"]:
            issues.append(f"too dark: mean_gray={brightness:.1f} < {cnf.CONFIG['brightness_low']:g}")
        if brightness > cnf.CONFIG["brightness_high"]:
            issues.append(f"too bright: mean_gray={brightness:.1f} > {cnf.CONFIG['brightness_high']:g}")
        if min(width, height) < cnf.CONFIG["min_resolution"]:
            issues.append(
                f"low resolution: {width}x{height}, short side < {cnf.CONFIG['min_resolution']}px"
            )

        metrics = {
            "blur_score": blur_score,
            "brightness": brightness,
            "width": int(width),
            "height": int(height),
        }
        return (len(issues) == 0), issues, metrics
    
class Face:
    @staticmethod
    def initMesh( # Init a mesh 
        staticImg: bool = True,
        maxFaces: int = 1,
        refineLandmarks: bool = True,
        minDetectionConfidence: float = 0.5,
        minTrackingConfidence: float = 0.5,
    ):
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=staticImg,
            max_num_faces=maxFaces,
            refine_landmarks=refineLandmarks,
            min_detection_confidence=minDetectionConfidence,
            min_tracking_confidence=minTrackingConfidence,
        )
    @staticmethod
    @contextmanager
    def meshSession(
        face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None
    ) -> Iterator:

        owns = face_mesh is None

        if face_mesh is None:
            face_mesh = Face.initMesh(
                staticImg=cnf.CONFIG["static_image_mode"],
                maxFaces=cnf.CONFIG["max_num_faces"],
                refineLandmarks=cnf.CONFIG["refine_landmarks"],
                minDetectionConfidence=cnf.CONFIG["min_detection_confidence"],
                minTrackingConfidence=cnf.CONFIG["min_tracking_confidence"],
            )

        try:
            yield face_mesh
        finally:
            if owns:
                face_mesh.close()
    @staticmethod
    def findLargest(face_landmarks_list: Sequence): # Find the largest face in a photo
        best, best_area = None, -1.0

        for face in face_landmarks_list:
            xs = [lm.x for lm in face.landmark]
            ys = [lm.y for lm in face.landmark]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))

            if area > best_area:
                best, best_area = face, area

        return best
    @staticmethod
    def extractLandmarks( # Find all landmarks
        image_rgb: np.ndarray,
        face_mesh: "mp.solutions.face_mesh.FaceMesh",
    ) -> Dict[str, Any]:

        result = {
            "face_found": False,
            "landmarks_norm": None,
            "landmarks_px": None,
            "n_faces": 0,
            "error": None,
        }

        if image_rgb is None or image_rgb.size == 0:
            result["error"] = "empty image"
            return result

        try:
            outputs = face_mesh.process(image_rgb)
        except Exception as exc:
            result["error"] = f"facemesh failure: {exc}"
            print("FaceMesh failed")
            return result

        multi = list(outputs.multi_face_landmarks or [])
        result["n_faces"] = len(multi)

        if not multi:
            result["error"] = "no face detected"
            return result

        chosen = Face.findLargest(multi)
        height, width = image_rgb.shape[:2]
        landmarks = chosen.landmark

        result["landmarks_norm"] = [
            (lm.x, lm.y, lm.z) for lm in landmarks
        ]

        result["landmarks_px"] = [
            (int(round(lm.x * width)), int(round(lm.y * height)))
            for lm in landmarks
        ]

        result["face_found"] = True
        return result
    @staticmethod
    def crop(
        image_rgb: np.ndarray,
        landmarks_px: Sequence[Sequence[float]],
        margin: Optional[float] = None,
        ) -> Optional[np.ndarray]:
   
        margin = cnf.CONFIG["crop_margin"] if margin is None else margin
        pts = np.asarray(landmarks_px, dtype=float)[:, :2]
        height, width = image_rgb.shape[:2]

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        box_w, box_h = x_max - x_min, y_max - y_min

        x0 = int(max(0, math.floor(x_min - margin * box_w)))
        y0 = int(max(0, math.floor(y_min - margin * box_h)))
        x1 = int(min(width, math.ceil(x_max + margin * box_w)))
        y1 = int(min(height, math.ceil(y_max + margin * box_h)))
        if x1 - x0 < 2 or y1 - y0 < 2:
           print("Degenerate crop box (%dx%d) - skipping crop.", x1 - x0, y1 - y0)
           return None

        return image_rgb[y0:y1, x0:x1].copy()
    @staticmethod
    def align(
        image_bgr: np.ndarray,
        landmarks_px: Sequence[Sequence[float]],
        output_size: Optional[Tuple[int, int]] = None,
        eye_target_x: Tuple[float, float] = (0.35, 0.65),
        eye_target_y: float = 0.40,
        ) -> Optional[np.ndarray]:
        output_size = output_size or cnf.CONFIG["align_output_size"]
        out_w, out_h = output_size
        pts = np.asarray(landmarks_px, dtype=float)[:, :2]

        src_right = pts[list(cnf.EYE_RIGHT_IDXS)].mean(axis=0)  # appears image-left (D9)
        src_left = pts[list(cnf.EYE_LEFT_IDXS)].mean(axis=0)    # appears image-right
        dst_right = np.array([eye_target_x[0] * out_w, eye_target_y * out_h])
        dst_left = np.array([eye_target_x[1] * out_w, eye_target_y * out_h])

        src_vec = src_left - src_right
        src_len = float(np.linalg.norm(src_vec))
        if src_len <= 0:
            print("Degenerate eye geometry (zero-length eye line); cannot align.")
            return None
        scale = float(np.linalg.norm(dst_left - dst_right)) / src_len

        angle = math.atan2(src_vec[1], src_vec[0])           # current eye-line angle
        cos_a, sin_a = math.cos(-angle), math.sin(-angle)    # rotate eye line onto +x axis
        rotation = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        src_mid = (src_right + src_left) / 2.0
        dst_mid = (dst_right + dst_left) / 2.0
        translation = dst_mid - rotation @ src_mid
        matrix = np.column_stack([rotation, translation])    # 2x3 affine

        return cv2.warpAffine(
            image_bgr, matrix, (out_w, out_h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
        )

    
class Geometric:
    @staticmethod
    def dist(p: Sequence[float], q: Sequence[float]) -> float:
        return float(math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))) # Distance between two points
    @staticmethod
    def eyeAspect(points: Sequence[Sequence[float]], eye: Dict[str, int]) -> float: # Openness of an eye
        # Borrowed from Felipe
        # Uses 6 points around an eye to infer how open it is
        # EAR = frac{|p2,p6| + |p3,p5|}{2|p1 + p4|}

        p1, p2, p3, p4, p5, p6 = (points[eye[k]] for k in ("p1", "p2", "p3", "p4", "p5", "p6"))
        vertical = Geometric.dist(p2, p6) + Geometric.dist(p3, p5)
        horizontal = Geometric.dist(p1, p4)
        if horizontal <= 0:
            return float("nan")
        return vertical / (2.0 * horizontal)
    @staticmethod
    def computeGeom(landmarks_px: Sequence[Sequence[float]]) -> Dict[str, float]:
        pts = [(float(p[0]), float(p[1])) for p in landmarks_px] # Convert points into a list of coordinate tuples

        if len(pts) <= cnf.MAX_USED_INDEX: # Not enough points
            raise ValueError(
                f"expected >= {cnf.MAX_USED_INDEX + 1} FaceMesh landmarks, got {len(pts)}"
            )

        eye_right_center = np.mean([pts[i] for i in cnf.EYE_RIGHT_IDXS], axis=0)
        eye_left_center = np.mean([pts[i] for i in cnf.EYE_LEFT_IDXS], axis=0)
        iod = float(np.linalg.norm(eye_left_center - eye_right_center))

        def by_iod(value: float) -> float:
            return value / iod if iod > 0 else float("nan")

        corner_right, corner_left = pts[cnf.MOUTH_CORNER_RIGHT], pts[cnf.MOUTH_CORNER_LEFT]
        mouth_width = Geometric.dist(corner_right, corner_left)
        mouth_height = Geometric.dist(pts[cnf.LIP_UPPER_INNER], pts[cnf.LIP_LOWER_INNER])

        lip_center_y = (pts[cnf.LIP_UPPER_INNER][1] + pts[cnf.LIP_LOWER_INNER][1]) / 2.0
        corners_mean_y = (corner_right[1] + corner_left[1]) / 2.0
        mouth_curvature = (
            (corners_mean_y - lip_center_y) / mouth_width if mouth_width > 0 else float("nan")
        )

        return {
            "iod_px": iod,
            "ear_left": Geometric.eyeAspect(pts, cnf.EYE_LEFT),
            "ear_right": Geometric.eyeAspect(pts, cnf.EYE_RIGHT),
            "mar": mouth_height / mouth_width if mouth_width > 0 else float("nan"),
            "mouth_width": by_iod(mouth_width),
            "mouth_height": by_iod(mouth_height),
            "brow_eye_distance_left": by_iod(Geometric.dist(pts[cnf.BROW_LEFT], eye_left_center)),
            "brow_eye_distance_right": by_iod(Geometric.dist(pts[cnf.BROW_RIGHT], eye_right_center)),
            "mouth_curvature": mouth_curvature,
            "nose_to_mouth_corner_left": by_iod(Geometric.dist(pts[cnf.NOSE_TIP], corner_left)),
            "nose_to_mouth_corner_right": by_iod(Geometric.dist(pts[cnf.NOSE_TIP], corner_right)),
            "face_width": by_iod(Geometric.dist(pts[cnf.FACE_SIDE_RIGHT], pts[cnf.FACE_SIDE_LEFT])),
            "face_height": by_iod(Geometric.dist(pts[cnf.FOREHEAD_TOP], pts[cnf.CHIN_BOTTOM])),
    }


class Landmark:
    @staticmethod
    def normalize( # Normalize the landmark points relative to where the face appears
        landmarks: Sequence[Sequence[float]],
        return_z: bool = False,
        z_values: Optional[Sequence[float]] = None,
        ) -> Optional[np.ndarray]:
        
        arr = np.asarray(landmarks, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            raise ValueError("landmarks must be an (N, 2) or (N, 3) array-like with N > 0")

        xy = arr[:, :2]
        origin = xy[list(cnf.ALL_EYE_IDXS)].mean(axis=0)                       # mid-eyes point
        eye_right_center = xy[list(cnf.EYE_RIGHT_IDXS)].mean(axis=0)
        eye_left_center = xy[list(cnf.EYE_LEFT_IDXS)].mean(axis=0)
        scale = float(np.linalg.norm(eye_left_center - eye_right_center))  # IOD

        if scale <= 0:
            print("Degenerate face scale (IOD=0); landmarks cannot be normalized.")
            return None

        normalized = (xy - origin) / scale
        if return_z:
            if arr.shape[1] >= 3:
                z = arr[:, 2]
            elif z_values is not None:
                z = np.asarray(z_values, dtype=float)
            else:
                raise ValueError(
                    "return_z=True requires (N, 3) landmarks or explicit z_values "
                    "(in the same pixel scale as x/y)"
                )
            if z.shape[0] != arr.shape[0]:
                raise ValueError("z_values length does not match the number of landmarks")
        
            normalized = np.column_stack([normalized, (z - z.mean()) / scale])
        
        return normalized
    


class ImgData:
    @staticmethod
    def parts(path: str, root: Optional[str]):
        p = Path(path)
        if root is None:
            return tuple(p.parts)
        try:
            return tuple(p.relative_to(Path(root)).parts)
        except ValueError:
            try:
                return tuple(p.resolve().relative_to(Path(root).resolve()).parts)
            except (ValueError, OSError):
                return tuple(p.parts)
    @staticmethod
    def subject(filename: str) -> Optional[str]:
        stem = Path(filename).stem
        token = re.split(r"[ _\-.]+", stem, maxsplit=1)[0] if stem else ""
        if not token or token.lower() in cnf.GENERIC_SUBJECT_TOKENS or token.isdigit():
            return None
        return token
    @staticmethod
    def parseSrc(path: str, root: Optional[str] = None) -> Optional[str]:
        parts = ImgData.parts(path, cnf.CONFIG["image_dir"] if root is None else root)
        dirs = parts[:-1]
        return dirs[0] if dirs else None
    @staticmethod
    def parseId(path: str, root: Optional[str] = None) -> Optional[str]:
        parts = ImgData.parts(path, cnf.CONFIG["image_dir"] if root is None else root)
        dirs = parts[:-1]
        non_label_dirs = [d for d in dirs if d.lower() not in cnf.CLASS_ALIASES]
        if len(non_label_dirs) >= 2:
            return non_label_dirs[-1]
        if len(non_label_dirs) == 1 and len(dirs) >= 2:
            return ImgData.subject(parts[-1])  # only source + label folders present
        return ImgData.subject(parts[-1])
    @staticmethod
    def parseLbl(path: str, root: Optional[str] = None) -> Optional[str]:
        parts = ImgData.parts(path, cnf.CONFIG["image_dir"] if root is None else root)
        for part in parts[:-1]:
            label = cnf.CLASS_ALIASES.get(part.lower())
            if label:
                return label
        tokens = re.split(r"[ _\-.]+", Path(parts[-1]).stem.lower())
        for token in tokens:
            label = cnf.CLASS_ALIASES.get(token)
            if label:
                return label
        return None
    @staticmethod
    def parse_metadata(path: str, root: Optional[str] = None) -> Dict[str, Optional[str]]:
        return {
            "image_path": str(path),
            "source": ImgData.parseSrc(path, root),
            "subject_id": ImgData.parseId(path, root),
            "label": ImgData.parseLbl(path, root),
        }


    
class Obtain:
    @staticmethod
    def safeCrop(path: str, root: str) -> str: # Eliminate collisions
        p = Path(path)
        try:
            rel = p.relative_to(Path(root))
        except ValueError:
            rel = p
        flat = "_".join(rel.with_suffix("").parts)
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", flat)
    @staticmethod
    def save(image_bgr: np.ndarray, landmarks_px: Sequence[Sequence[float]], # Save the cleaned up images
                crops_dir: Path, path: str, root: str) -> None:

        raw_dir, aligned_dir = crops_dir / "raw", crops_dir / "aligned"
        raw_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        stem = Obtain.safeCrop(path, root)
        crop = Face.crop(image_bgr, landmarks_px)
        if crop is not None:
            cv2.imwrite(str(raw_dir / f"{stem}.jpg"), crop)
        aligned = Face.align(image_bgr, landmarks_px)
        if aligned is not None:
            cv2.imwrite(str(aligned_dir / f"{stem}.jpg"), aligned)
    @staticmethod
    def process(
    path: str,
    image_rgb: Optional[np.ndarray],
    face_mesh: "mp.solutions.face_mesh.FaceMesh",
    root: str,
    crops_dir: Optional[Path] = None,
    include_z: bool = False,
    ) -> Dict[str, Any]:
   
        record: Dict[str, Any] = {
            "image_path": str(path),
            "source": ImgData.parseSrc(path, root),
            "subject_id": ImgData.parseId(path, root),
            "label": ImgData.parseLbl(path, root),
            "width": np.nan,
            "height": np.nan,
            "blur_score": np.nan,
            "brightness": np.nan,
            "face_found": False,
            "n_landmarks": 0,
            "n_faces": 0,
            "rejected_reason": "",
            "error": None,
            "_landmarks_unit": None,
            "geometry": None,
        }
        if image_rgb is None:  # unreadable (load_images already logged it)
            record["rejected_reason"] = "unreadable image"
            record["error"] = "unreadable image"
            return record
        
        ok, issues, metrics = Image.validate(image_rgb)
        record.update(metrics)
        if not ok:
            record["rejected_reason"] = "; ".join(issues)
            record["error"] = f"rejected at ingestion: {record['rejected_reason']}"
            return record
        
        landmarks = Face.extractLandmarks(image_rgb, face_mesh)
        record["n_faces"] = landmarks["n_faces"]
        record["error"] = landmarks["error"]
        if not landmarks["face_found"]:
            record["rejected_reason"] = "no face detected"
            return record
        
        record["face_found"] = True
        record["n_landmarks"] = len(landmarks["landmarks_px"])
        try:
            record["geometry"] = Geometric.computeGeom(landmarks["landmarks_px"])
            # z handling (D1): FaceMesh's native z is roughly x-normalized, so scale it
            # to pixel units (z * width) before normalization gives it IOD units too.
            z_pixel = (
                [z * record["width"] for _, _, z in landmarks["landmarks_norm"]]
                if include_z
                else None
            )
            record["_landmarks_unit"] = Landmark.normalize(
                landmarks["landmarks_px"], return_z=include_z, z_values=z_pixel
            )
        except (ValueError, IndexError) as exc:
            record["error"] = f"feature extraction failed: {exc}"
            print(f"{path}: {exc}")
            return record
        if record["_landmarks_unit"] is None:
            record["error"] = "landmark normalization failed (degenerate face scale)"
        
        if crops_dir is not None:
            Obtain.save(image_rgb, landmarks["landmarks_px"], crops_dir, path, root)
        return record
    @staticmethod
    def iterProcessed(
    image_dir: str,
    face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    root: Optional[str] = None,
    crops_dir: Optional[Path] = None,
    include_z: bool = False,
    extensions: Sequence[str] = cnf.CONFIG["extensions"],
    ):

        parse_root = Path(image_dir if root is None else root)
        with Face.meshSession(face_mesh) as fm:
            for path, image in tqdm.tqdm(Image.readAll(image_dir, extensions), desc="Processing images"):
                try:
                    yield Obtain.process(
                        path, image, fm, root=parse_root,
                        crops_dir=crops_dir, include_z=include_z,
                    )
                except Exception as exc:  # absolute last-resort guard: keep the run alive
                    
                    record = ImgData.parse_metadata(path, parse_root)
                    record.update({
                        "width": np.nan, "height": np.nan, "blur_score": np.nan,
                        "brightness": np.nan, "face_found": False, "n_landmarks": 0,
                        "n_faces": 0, "rejected_reason": f"processing error: {exc}",
                        "error": f"processing error: {exc}",
                        "_landmarks_unit": None, "geometry": None,
                    })
                    yield record
    @staticmethod
    def buildManifest(
    image_dir: str,
    face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    output_csv: Optional[str] = None,
    root: Optional[str] = None,
    ):
        
        output_csv = Path(cnf.CONFIG["output_dir"]) / "manifest.csv" if output_csv is None else Path(output_csv)
        rows = [
            {col: record.get(col) for col in cnf.MANIFEST_COLUMNS}
            for record in Obtain.iterProcessed(image_dir, face_mesh=face_mesh, root=root)
        ]
        manifest = pd.DataFrame(rows, columns=cnf.MANIFEST_COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(output_csv, index=False)  # overwrite => idempotent (D8)
        print(f"Wrote {output_csv} ({len(manifest)} rows)")
        return manifest
    @staticmethod
    def buildGeomTable(
    image_dir: str,
    face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    output_csv: Optional[str] = None,
    root: Optional[str] = None,
    ):

        output_csv = (
            Path(cnf.CONFIG["output_dir"]) / "geometric_features.csv"
            if output_csv is None else Path(output_csv)
        )
        rows: List[Dict[str, Any]] = []
        for record in Obtain.iterProcessed(image_dir, face_mesh=face_mesh, root=root):
            if record["rejected_reason"]:
                continue
            row: Dict[str, Any] = {col: record.get(col) for col in cnf.GEOMETRIC_META_COLUMNS}
            geometry = record.get("geometry") or {}
            row["iod_px"] = geometry.get("iod_px", np.nan)
            for feature in cnf.GEOMETRIC_FEATURE_ORDER:
                row[feature] = geometry.get(feature, np.nan)
            row["error"] = record.get("error")
            rows.append(row)

        table = pd.DataFrame(rows, columns=cnf.GEOMETRIC_COLUMNS)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv} ({len(table)} rows)")
        return table
    @staticmethod
    def landmarkIndex(landmark_indices: Sequence[int], include_z: bool) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for index in landmark_indices:
            rows.append({"column_name": f"x_{index}", "face_mesh_index": index, "axis": "x"})
            rows.append({"column_name": f"y_{index}", "face_mesh_index": index, "axis": "y"})
            if include_z:
                rows.append({"column_name": f"z_{index}", "face_mesh_index": index, "axis": "z"})
        return pd.DataFrame(rows, columns=["column_name", "face_mesh_index", "axis"])
    @staticmethod
    def buildLandmark(
    image_dir: str,
    face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    include_z: Optional[bool] = None,
    landmark_indices: Optional[Sequence[int]] = None,
    output_csv: Optional[str] = None,
    index_csv: Optional[str] = None,
    root: Optional[str] = None,
    ) -> pd.DataFrame:
        
        include_z = cnf.CONFIG["include_z"] if include_z is None else include_z
        landmark_indices = (
            cnf.CONFIG["landmark_indices"] if landmark_indices is None else landmark_indices
        )
        output_csv = (
            Path(cnf.CONFIG["output_dir"]) / "landmark_features.csv"
            if output_csv is None else Path(output_csv)
        )
        index_csv = (
            Path(cnf.CONFIG["output_dir"]) / "landmark_index.csv"
            if index_csv is None else Path(index_csv)
        )

        expected_count = 478 if cnf.CONFIG["refine_landmarks"] else 468
        indices_used = (
            list(range(expected_count)) if landmark_indices is None else list(landmark_indices)
        )

        rows: List[Dict[str, Any]] = []
        for record in Obtain.iterProcessed(
            image_dir, face_mesh=face_mesh, root=root, include_z=include_z
        ):
            if record["rejected_reason"]:
                continue  # gated (D10)
            row: Dict[str, Any] = {col: record.get(col) for col in cnf.LANDMARK_META_COLUMNS}
            unit = record.get("_landmarks_unit")
            if unit is not None:
                for index in indices_used:
                    if index < unit.shape[0]:
                        row[f"x_{index}"] = float(unit[index, 0])
                        row[f"y_{index}"] = float(unit[index, 1])
                        if include_z and unit.shape[1] > 2:
                            row[f"z_{index}"] = float(unit[index, 2])
            rows.append(row)

        table = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
        Obtain.landmarkIndex(indices_used, include_z).to_csv(index_csv, index=False)
        print(f"Wrote {index_csv}")
        return table
    @staticmethod
    def audit(
    manifest: pd.DataFrame,
    geometric_df: Optional[pd.DataFrame] = None,
    landmark_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        manifest = manifest.copy()
        manifest["rejected_reason"] = manifest["rejected_reason"].fillna("")
        rejected_mask = manifest["rejected_reason"] != ""
        n_total = len(manifest)
        n_rejected = int(rejected_mask.sum())
        accepted = manifest.loc[~rejected_mask]
        face_rate = float(accepted["face_found"].mean()) if len(accepted) else float("nan")

        summary: Dict[str, Any] = {
            "n_images": n_total,
            "n_accepted": n_total - n_rejected,
            "n_rejected": n_rejected,
            "face_detection_rate": face_rate,
            "images_per_source": manifest["source"].value_counts(dropna=False).to_dict(),
            "images_per_subject": manifest["subject_id"].value_counts(dropna=False).to_dict(),
            "rejection_reasons": manifest.loc[rejected_mask, "rejected_reason"].value_counts().to_dict(),
            "label_distribution": manifest["label"].value_counts(dropna=False).to_dict(),
            "label_distribution_accepted": accepted["label"].value_counts(dropna=False).to_dict(),
        }

        print("=" * 62)
        print("OBTAIN STAGE - DATASET SUMMARY")
        print("=" * 62)
        print(f"Images found           : {summary['n_images']}")
        print(f"Accepted (validated)   : {summary['n_accepted']}")
        print(f"Rejected at ingestion  : {summary['n_rejected']}")
        print(f"Face detection rate    : {face_rate:.1%} of accepted"
              if len(accepted) else "Face detection rate    : n/a (no accepted images)")
        print("-" * 62)
        print("Images per source:")
        for source, count in summary["images_per_source"].items():
            print(f"  {source!s:<30} {count}")
        print("Images per subject:")
        for subject, count in summary["images_per_subject"].items():
            print(f"  {subject!s:<30} {count}")
        print("Rejection reasons:")
        for reason, count in summary["rejection_reasons"].items():
            print(f"  {count:>5} x {reason}")
        print("Label distribution (accepted images):")
        for label, count in summary["label_distribution_accepted"].items():
            print(f"  {label!s:<30} {count}")
        print("-" * 62)
        if geometric_df is not None:
            expected = summary["n_accepted"]
            if len(geometric_df) != expected:
                print(f"WARNING: geometric_features rows = {len(geometric_df)}, expected {expected}")
            else:
                print(f"geometric_features rows = {len(geometric_df)} (ok)")
        if landmark_df is not None:
            expected = summary["n_accepted"]
            if len(landmark_df) != expected:
                print(f"WARNING: landmark_features rows = {len(landmark_df)}, expected {expected}")
            else:
                print(f"landmark_features rows  = {len(landmark_df)} (ok)")
        print("=" * 62)
        return summary

    @staticmethod
    def run(
    image_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_crops: Optional[bool] = None,
    include_z: Optional[bool] = None,
    face_mesh: Optional["mp.solutions.face_mesh.FaceMesh"] = None,
    extensions: Sequence[str] = cnf.CONFIG["extensions"],
    ) -> Dict[str, pd.DataFrame]:

        image_dir = Path(image_dir or cnf.CONFIG["image_dir"])
        output_dir = Path(output_dir or cnf.CONFIG["output_dir"])
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Input image directory not found: {image_dir.resolve()}")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_crops = cnf.CONFIG["save_crops"] if save_crops is None else save_crops
        include_z = cnf.CONFIG["include_z"] if include_z is None else include_z

        crops_dir = output_dir / "crops"
        if save_crops:
            if crops_dir.exists():
                shutil.rmtree(crops_dir)  # idempotency: no stale crops across runs (D8)
            crops_dir.mkdir(parents=True, exist_ok=True)

        records = list(
            Obtain.iterProcessed(
                image_dir, face_mesh=face_mesh, root=image_dir,
                crops_dir=crops_dir if save_crops else None,
                include_z=include_z, extensions=extensions,
            )
        )
        if not records:
            print(f"No images found under {image_dir} - empty tables will be written.")

        # --- manifest + rejection log -------------------------------------------
        manifest = pd.DataFrame(
            [{col: record.get(col) for col in cnf.MANIFEST_COLUMNS} for record in records],
            columns=cnf.MANIFEST_COLUMNS,
        )
        rejected_mask = manifest["rejected_reason"].fillna("") != ""
        rejected = manifest.loc[rejected_mask, cnf.REJECTED_COLUMNS].copy()

        # --- geometric table (Approach A) ----------------------------------------
        geom_rows: List[Dict[str, Any]] = []
        for record in records:
            if record["rejected_reason"]:
                continue
            row = {col: record.get(col) for col in cnf.GEOMETRIC_META_COLUMNS}
            geometry = record.get("geometry") or {}
            row["iod_px"] = geometry.get("iod_px", np.nan)
            for feature in cnf.GEOMETRIC_FEATURE_ORDER:
                row[feature] = geometry.get(feature, np.nan)
            row["error"] = record.get("error")
            geom_rows.append(row)
        geometric = pd.DataFrame(geom_rows, columns=cnf.GEOMETRIC_COLUMNS)

        # --- landmark table (Approach B) + index ---------------------------------
        expected_count = 478 if cnf.CONFIG["refine_landmarks"] else 468
        indices_used = list(range(expected_count)) if cnf.CONFIG["landmark_indices"] is None else list(cnf.CONFIG["landmark_indices"])
        lm_rows: List[Dict[str, Any]] = []
        for record in records:
            if record["rejected_reason"]:
                continue
            row = {col: record.get(col) for col in cnf.LANDMARK_META_COLUMNS}
            unit = record.get("_landmarks_unit")
            if unit is not None:
                for index in indices_used:
                    if index < unit.shape[0]:
                        row[f"x_{index}"] = float(unit[index, 0])
                        row[f"y_{index}"] = float(unit[index, 1])
                        if include_z and unit.shape[1] > 2:
                            row[f"z_{index}"] = float(unit[index, 2])
            lm_rows.append(row)
        landmarks_df = pd.DataFrame(lm_rows)
        landmark_index = Obtain.landmarkIndex(indices_used, include_z)

        manifest.to_csv(output_dir / "manifest.csv", index=False)
        rejected.to_csv(output_dir / "rejected_images.csv", index=False)
        geometric.to_csv(output_dir / "geometric_features.csv", index=False)
        landmarks_df.to_csv(output_dir / "landmark_features.csv", index=False)
        landmark_index.to_csv(output_dir / "landmark_index.csv", index=False)
        print(
            f"Obtain complete: {len(manifest)} manifest rows, {len(rejected)} rejected, {len(geometric)} geometric rows, {len(landmarks_df)} landmark rows",
        )
        return {
            "manifest": manifest,
            "rejected": rejected,
            "geometric": geometric,
            "landmarks": landmarks_df,
            "landmark_index": landmark_index,
        }

Obtain.run()