import unittest

from legalrag.preprocessing.revalidar_cache_docx import unchanged_except_docx


class SelectiveCacheGuardTests(unittest.TestCase):
    BASE = 'LIMIT = 42\ndef extraer_docx(data):\n    return data\ndef extraer_html(data):\n    return data\n'

    def test_docx_body_change_is_allowed(self):
        changed = self.BASE.replace('def extraer_docx(data):\n    return data',
                                    'def extraer_docx(data):\n    return data.strip()')
        self.assertTrue(unchanged_except_docx(self.BASE, changed))

    def test_global_or_other_extractor_change_is_rejected(self):
        self.assertFalse(unchanged_except_docx(self.BASE, self.BASE.replace('42', '43')))
        self.assertFalse(unchanged_except_docx(self.BASE, self.BASE + '\ndef other():\n    pass\n'))
        changed = self.BASE.replace('def extraer_html(data):\n    return data',
                                    'def extraer_html(data):\n    return data.strip()')
        self.assertFalse(unchanged_except_docx(self.BASE, changed))

    def test_missing_or_duplicate_docx_extractor_is_rejected(self):
        with self.assertRaises(ValueError):
            unchanged_except_docx(self.BASE, 'LIMIT = 42')
        with self.assertRaises(ValueError):
            unchanged_except_docx(self.BASE, self.BASE + '\ndef extraer_docx(data):\n    pass\n')


if __name__ == '__main__':
    unittest.main()
