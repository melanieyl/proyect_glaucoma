import os, re, json, tempfile, unittest, mimetypes, random, glob
import requests

API_URL = os.getenv("API_URL", "http://localhost:8088")
PREDICT = f"{API_URL}/predict"

TEST_IMG_ADV = os.getenv("TEST_IMG_ADV") 
TEST_DIR_NORMAL = os.getenv("TEST_DIR_NORMAL")
PATTERN_LIMIT = int(os.getenv("PATTERN_LIMIT", "50"))
PATTERN_EXPECT_MIN_NORMAL = float(os.getenv("PATTERN_EXPECT_MIN_NORMAL", "0.6"))

def _parse_api_response(text: str):
    try:
        obj = json.loads(text)
        return {
            "label": obj.get("label"),
            "prob": obj.get("confidence") or obj.get("prob"),
            "index": obj.get("index")
        }
    except Exception:
        pass
    lab = re.search(r"Predicci[a-zA-Z\u00f3\u00d3]*:\s*([^<]+)<", text)
    conf = re.search(r"Confianza:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    return {
        "label": lab.group(1).strip() if lab else None,
        "prob": float(conf.group(1)) / 100.0 if conf else None,
        "index": None
    }

def _post_image(path: str, timeout=30):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return requests.post(PREDICT, files={"file": (os.path.basename(path), f, mime)}, timeout=timeout)

class APISmokeTests(unittest.TestCase):
    def test_smoke_root(self):
        r = requests.get(API_URL, timeout=15)
        self.assertEqual(r.status_code, 200)
        self.assertTrue("Glaucoma" in r.text or "Clasificación" in r.text)

    def test_smoke_predict_without_file(self):
        r = requests.post(PREDICT, files={}, timeout=15)
        self.assertIn(r.status_code, (400, 415))

class APIOneShotTests(unittest.TestCase):
    def test_one_shot_advanced(self):
        if not TEST_IMG_ADV or not os.path.exists(TEST_IMG_ADV):
            self.skipTest("Define TEST_IMG_ADV con la ruta a una imagen 'Glaucoma_Advanced'.")
        r = _post_image(TEST_IMG_ADV)
        self.assertEqual(r.status_code, 200, r.text)
        parsed = _parse_api_response(r.text)
        self.assertIsNotNone(parsed["label"], f"Respuesta no parseable: {r.text}")
        self.assertEqual(parsed["label"], "Glaucoma_Advanced")

class APIEdgeTests(unittest.TestCase):
    def test_edge_invalid_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"not an image")
            tmp_path = tmp.name
        try:
            r = _post_image(tmp_path)
            self.assertEqual(r.status_code, 400, r.text)
        finally:
            os.remove(tmp_path)

    def test_edge_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            r = _post_image(tmp_path)
            self.assertGreaterEqual(r.status_code, 400, r.text)
        finally:
            os.remove(tmp_path)

class APIPatternTests(unittest.TestCase):
    def test_pattern_many_normals(self):
        if not TEST_DIR_NORMAL or not os.path.isdir(TEST_DIR_NORMAL):
            self.skipTest("Define TEST_DIR_NORMAL con carpeta de imágenes 'Normal'.")
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(TEST_DIR_NORMAL, e)))
        if not files:
            self.skipTest("No se encontraron imágenes en TEST_DIR_NORMAL.")
        random.shuffle(files)
        files = files[:PATTERN_LIMIT]

        total = 0
        normals = 0
        for p in files:
            r = _post_image(p)
            if r.status_code != 200:
                continue
            parsed = _parse_api_response(r.text)
            if parsed["label"]:
                total += 1
                if parsed["label"] == "Normal":
                    normals += 1

        self.assertGreater(total, 0, "No se obtuvieron predicciones válidas.")
        ratio = normals / total
        self.assertGreaterEqual(
            ratio, PATTERN_EXPECT_MIN_NORMAL,
            f"Solo {ratio:.2%} 'Normal' (< {PATTERN_EXPECT_MIN_NORMAL:.0%})."
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
