import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat import Answer
from chat_batch import grade


class BatchGradingTests(unittest.TestCase):
    def test_six_correct_answers(self):
        for q, a in [('日本の首都は？','日本の首都は東京です。'), ('犬は動物ですか？','はい、犬は動物です。'),
                     ('1+1はいくつ？','2です。'), ('3と5はどちらが大きい？','5の方が大きいです。'),
                     ('800円の30%引きはいくら？','支払額は560円です。'), ('14時から30分後は何時？','14時30分です。')]:
            # Explicitly cover natural copula on the comparison answer too.
            self.assertEqual(grade(q, Answer(a,'eos',[3]))[0], '正解', (q,a))

    def test_false_positives_and_contradictions(self):
        for q, a in [('1+1はいくつ？','2はいくつ？'), ('1+1はいくつ？','答えは2ではなく3です。'),
                     ('800円の30%引きはいくら？','560円ではありません。'),
                     ('日本の首都は？','東京は都市ですが首都は大阪です。'),
                     ('犬は動物ですか？','動物は動物です。'),
                     ('3と5はどちらが大きい？','5より3が大きいです。')]:
            self.assertEqual(grade(q, Answer(a,'eos',[3]))[0], '不正解')

    def test_natural_uncertain_and_repetition(self):
        self.assertEqual(grade('おはよう', Answer('おはようございます。','eos',[3]))[0], '自然な回答')
        self.assertEqual(grade('最近どう？', Answer('いろいろありました。','eos',[3]))[0], '要確認')
        self.assertEqual(grade('明日は何曜日？', Answer('月曜日です。','eos',[3]))[0], '要確認')
        self.assertEqual(grade('犬は動物ですか？', Answer('犬は動物です。'*3,'eos',[3]))[0], '不正解')


if __name__ == '__main__':
    unittest.main()
