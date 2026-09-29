import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from legalrag.ingestion.corregir_manifest import correct_rad_numbers, resolve_inside
from legalrag.preprocessing.preparar_corpus import prepare_one


class CorpusPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def asset(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {'archivo': relative, 'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest(), 'url': 'https://example.org/legal'}

    def test_rad_requires_matching_header_and_is_idempotent(self):
        entry = self.asset('sentencia/000.doc', b'preserved binary')
        entry['texto_derivado'] = self.asset('sentencia/000.txt', b'Proceso No 27283\nTEXTO')
        doc = {'doc_id': 'sentencia_csj_rad27283_2007', 'numero': '27283(01-08-07)',
               'archivos_raw': [entry]}
        changes = correct_rad_numbers([doc], self.root)
        self.assertEqual(doc['numero'], '27283')
        self.assertEqual(doc['numero_original_manifest'], '27283(01-08-07)')
        self.assertEqual(len(changes), 1)
        self.assertEqual(correct_rad_numbers([doc], self.root), [])
        doc['numero'] = '99999(01-08-07)'
        self.assertEqual(correct_rad_numbers([doc], self.root), [])

    def test_bad_hash_cannot_support_a_correction(self):
        entry = self.asset('sentencia/000.doc', b'preserved binary')
        entry['texto_derivado'] = self.asset('sentencia/000.txt', b'Proceso No 27283')
        (self.root / 'sentencia/000.txt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            correct_rad_numbers([{'doc_id': 'sentencia_csj_rad27283_2007',
                                  'numero': '27283(01-08-07)', 'archivos_raw': [entry]}], self.root)

    def test_paths_cannot_escape_raw(self):
        with self.assertRaises(ValueError):
            resolve_inside(self.root, '../elsewhere')

    def test_omitted_source_is_not_marked_searchable(self):
        text = 'DECRETO 123 DE 2000. Esta compilación no incluye el texto de este decreto.'
        entry = self.asset('decreto_123_2000/000.html', ('<html><body><p>' + text + '</p></body></html>').encode())
        doc = {'doc_id': 'decreto_123_2000', 'tipo': 'decreto', 'url': entry['url'],
               'archivos_raw': [entry], 'redistribuir_raw': True}
        result = prepare_one(doc, self.root, self.root / 'prepared', 'test')
        self.assertFalse(result['apta_para_busqueda'])
        self.assertIn('fuente_declara_omision_del_texto', result['pendientes_preparacion'])
        self.assertTrue((self.root / 'decreto_123_2000/000.html').exists())
        prepared = self.root / 'prepared' / result['texto_archivo']
        self.assertEqual(hashlib.sha256(prepared.read_bytes()).hexdigest(), result['sha256_texto'])


if __name__ == '__main__':
    unittest.main()
