"""Small regression checks for the shipped scope and original-photo shadow estimator."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NODE = ROOT / 'custom_nodes/product_studio_rgb240'
spec = importlib.util.spec_from_file_location('rgb240_source_shadow', NODE / 'source_shadow.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleTests(unittest.TestCase):
    def test_only_final_nodes_exported(self):
        tree = ast.parse((NODE / '__init__.py').read_text())
        extension = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ProductStudioRGB240Extension')
        returned = next(n for n in ast.walk(extension) if isinstance(n, ast.Return))
        exported = [n.id for n in returned.value.elts]
        self.assertEqual(exported, json.loads((ROOT / 'SOURCE_LOCK.json').read_text())['expected_node_ids'])
        self.assertFalse((NODE / 'studio_grounded.py').exists())
        self.assertFalse((NODE / 'studio_locked.py').exists())

    def test_ui_and_api_links_agree(self):
        api = json.loads((ROOT / 'workflows/rgb240_api.json').read_text())
        ui = json.loads((ROOT / 'workflows/rgb240_ui.json').read_text())
        nodes = {str(n['id']): n for n in ui['nodes']}
        links = {n[0]: n for n in ui['links']}
        self.assertEqual(len(api), 7)
        for node_id, item in api.items():
            self.assertEqual(nodes[node_id]['type'], item['class_type'])
            for name, value in item['inputs'].items():
                if isinstance(value, list):
                    inp = next(n for n in nodes[node_id]['inputs'] if n['name'] == name)
                    link = links[inp['link']]
                    self.assertEqual([str(link[1]), link[2]], value)
                    self.assertIn(link[0], nodes[value[0]]['outputs'][value[1]]['links'])
        self.assertEqual(api['1']['class_type'], 'LoadImage')
        self.assertEqual(api['7']['class_type'], 'SaveImage')

    def test_model_manifest_complete(self):
        manifest = json.loads((ROOT / 'models/manifest.json').read_text())
        self.assertEqual(len(manifest['files']), 5)
        self.assertEqual(manifest['total_bytes'], sum(x['bytes'] for x in manifest['files']))
        for item in manifest['files']:
            self.assertEqual(len(item['sha256']), 64)
            self.assertTrue(item['url'].startswith('https://'))
            self.assertNotIn('local_source', item)


class ShadowTests(unittest.TestCase):
    def setUp(self):
        h, w = 620, 900
        self.alpha = np.zeros((h, w), np.float32)
        self.alpha[100:400, 200:650] = 1
        self.parts = self.alpha.astype('uint8')[None]
        self.excluded = np.zeros_like(self.alpha)
        self.plain = np.full((h, w, 3), .78, np.float32)
        self.plain[self.alpha > 0] = [.16, .10, .06]
        y, x = np.mgrid[:h, :w]
        cast = .32 * np.exp(-((x-575)/75)**2 - ((y-416)/18)**2)
        cast *= self.alpha == 0
        self.photo = np.full_like(self.plain, .78) * (1-cast[..., None])
        self.photo[self.alpha > 0] = [.16, .10, .06]

    def test_blank_floor_does_not_create_outline(self):
        shadow, _ = module.source_shadow(self.plain, self.alpha, self.excluded, self.parts)
        self.assertLess(float(shadow[self.alpha == 0].max()), .005)

    def test_directional_shadow_keeps_illuminated_side_clear(self):
        shadow, _ = module.source_shadow(self.photo, self.alpha, self.excluded, self.parts)
        self.assertGreater(float(shadow[412:435, 530:600].mean()), .025)
        self.assertLess(float(shadow[410:430, 240:340].max()), .015)

    def test_excluded_paper_print_is_not_a_shadow(self):
        card = self.photo.copy()
        card[410:450, 300:370] = .95
        card[420:426, 305:365] = .05
        excluded = self.excluded.copy()
        excluded[407:454, 296:374] = 1
        shadow, _ = module.source_shadow(card, self.alpha, excluded, self.parts)
        self.assertLess(float(shadow[414:444, 302:368].max()), .015)


if __name__ == '__main__':
    unittest.main()
