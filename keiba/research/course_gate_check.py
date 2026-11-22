"""芝のコース別に、枠の位置（1・2枠＝内、3〜6枠＝中、7・8枠＝外）の成績を市場と比べ、枠替わりフラグを照合する（研究のみ）。

使い方:
  python3 -m keiba.research.course_gate_check [keiba/reports/course_gate_check.md]

- 対象: keiba/data の JRA 平地芝重賞（2016/10〜）。頭数 engine.SMALL_FIELD 以下・オッズ欠け・1着が1頭でないレースは除く
- コース: 競馬場＋距離＋外回り/内回り（キャッシュ済みの結果ページの「芝右 外1600m」から。表記が無いものは区別しない）
- 期間: 学習＝〜2023年、評価＝2024年〜（年で分ける）。表は戦績が揃う 2021/10〜 を主に、2016/10〜2021/9 を独立の再現確認に使う
- 市場の見込み: form_check.market と同じ（勝率＝1/オッズの正規化、3着内＝単勝市場の λ で補正した Benter 型）。
  z＝(実数−見込み)/√Σp(1−p)。「均等」は 1/頭数（3着内は 3/頭数）。枠は抽選なので、実数÷均等 は枠そのものの効果
  （能力の偏りは誤差として入る）、市場÷均等 は市場が枠をどれだけ織り込んでいるか、実数÷市場 は市場の織り込み残し
- 枠替わり: 前走の枠（馬の戦績の枠番。前走が8頭以下なら馬番＝枠）を同じ区分にし、区分が変わった馬（芝の今回）を
  engine の turf_to_inner（+2）・turf_to_outer（−3）に当てはめる。温度 engine.SOFTMAX_TEMP で勝率の倍率 exp(点/温度) に換算し、
  (a) 中→外 などの組の 実数÷均等 の比（枠そのものの効果）と、(b) 市場にその倍率を掛けたときの評価年の対数尤度（市場の織り込み残し）と比べる
"""
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from keiba import engine
from keiba.research import history
from keiba.research.form_check import GRADED, JRA, market

OUT = Path(__file__).resolve().parents[1] / 'reports' / 'course_gate_check.md'
MAIN_FROM = '2021-10-01'
EVAL_FROM = '2024-01-01'
MIN_RACES = 12  # 2021/10〜 にこのレース数以上あるコースを個別に出す
ZONES = ('内', '中', '外')


def _flag_points():
    """engine の点数表から turf_to_inner / turf_to_outer を探す（表の名前に依存しない）。"""
    pts = {}
    for v in vars(engine).values():
        if isinstance(v, dict):
            for k in ('turf_to_inner', 'turf_to_outer'):
                if isinstance(v.get(k), (int, float)) and not isinstance(v.get(k), bool):
                    pts[k] = v[k]
    return pts


FLAGS = _flag_points()
T = engine.SOFTMAX_TEMP


def zone(gate):
    return engine.gate_zone_of(gate)


def loop(race_id):
    p = history.CACHE / f'race_{race_id}.html'
    if not p.exists():
        return ''
    m = re.search(r'<dl class="racedata[^"]*">(.*?)</dl>', p.read_text(encoding='utf-8'), re.S)
    t = history._text(m.group(1)) if m else ''
    m = re.search(r'芝[左右直]*\s*(外|内)', t)
    return m.group(1) if m else ''


def course_of(r):
    lp = loop(r['race_id'])
    return f"{r['place']}芝{r['distance']}{'外' if lp == '外' else '内' if lp == '内' and r['place'] in ('京都', '阪神', '新潟') else ''}"


_prev_cache = {}


def prev_run(horse_id, before):
    """判断日前の直近の出走の (枠, 頭数, 日付, 開催)。戦績がキャッシュに無ければ None、枠が読めなければ ('?',...)。"""
    if horse_id not in _prev_cache:
        p = history.CACHE / f'horse_{horse_id}.json'
        rows = []
        if p.exists():
            for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', json.loads(p.read_text(encoding='utf-8'))['data'], re.S):
                c = [history._text(x) for x in re.findall(r'<td[^>]*>(.*?)</td>', tr, re.S)]
                if len(c) >= 15:
                    rows.append(c)
            _prev_cache[horse_id] = rows
        else:
            _prev_cache[horse_id] = None
    rows = _prev_cache[horse_id]
    if rows is None:
        return None
    for c in rows:
        d = c[0].replace('/', '-')
        if d < before:
            if c[11] in ('取', '除'):
                continue
            field = int(c[6]) if c[6].isdigit() else None
            num = int(c[8]) if c[8].isdigit() else None
            gate = int(c[7]) if c[7].isdigit() else None
            if field and field <= 8 and num:
                gate = num
            return (gate, field, d, c[1]) if gate else ('?', field, d, c[1])
    return ('?', None, None, None)  # 初出走（重賞では実質ない）


def load():
    races = {}
    for p in GRADED:
        for r in json.loads(p.read_text(encoding='utf-8')):
            races[r['race_id']] = r
    out = []
    for r in sorted(races.values(), key=lambda r: r['date']):
        if r['surface'] != '芝' or r['place'] not in JRA:
            continue
        hs = [h for h in r['horses'] if h['rank'] not in ('取', '除') and h['odds']]
        if len(hs) <= engine.SMALL_FIELD or sum(h['rank'] == '1' for h in hs) != 1 or len(hs) != len(
                [h for h in r['horses'] if h['rank'] not in ('取', '除')]):
            continue
        pw, pp = market({**r, 'horses': hs})
        n = len(hs)
        rows = []
        for h in hs:
            rank = int(h['rank']) if h['rank'].isdigit() else 99
            pr = prev_run(h['horse_id'], r['date']) if h['horse_id'] else None
            pz = None if pr is None or pr[0] == '?' else zone(pr[0])
            rows.append({'num': h['num'], 'zone': zone(h['waku']), 'prev_zone': pz, 'win': rank == 1, 'top3': rank <= 3,
                         'pw': pw[h['num']], 'pp': pp[h['num']], 'uw': 1 / n, 'up': 3 / n})
        out.append({'race_id': r['race_id'], 'name': r['name'], 'date': r['date'], 'year': int(r['date'][:4]),
                    'course': course_of(r), 'n': n, 'rows': rows})
    return out


def period(r):
    if r['date'] < MAIN_FROM:
        return '2016/10〜2021/9'
    return '学習（2021/10〜2023）' if r['date'] < EVAL_FROM else '評価（2024〜）'


class Acc:
    __slots__ = ('n', 'w', 't', 'ew', 'et', 'vw', 'vt', 'uw', 'ut')

    def __init__(self):
        self.n = self.w = self.t = 0
        self.ew = self.et = self.vw = self.vt = self.uw = self.ut = 0.0

    def add(self, x):
        self.n += 1
        self.w += x['win']
        self.t += x['top3']
        self.ew += x['pw']
        self.et += x['pp']
        self.vw += x['pw'] * (1 - x['pw'])
        self.vt += x['pp'] * (1 - x['pp'])
        self.uw += x['uw']
        self.ut += x['up']

    def zw(self):
        return (self.w - self.ew) / math.sqrt(self.vw) if self.vw else 0.0

    def zt(self):
        return (self.t - self.et) / math.sqrt(self.vt) if self.vt else 0.0


def _r(a, b):
    return f'{a / b:.2f}' if b else '—'


def zone_rows(acc, label=''):
    out = []
    for z in ZONES:
        a = acc.get(z)
        if not a or not a.n:
            continue
        out.append(f'| {label}{z} | {a.n} | {a.w} | {a.ew:.1f} | {_r(a.w, a.ew)} | {a.zw():+.1f} | {a.t} | {a.et:.1f} | '
                   f'{_r(a.t, a.et)} | {a.zt():+.1f} | {_r(a.w, a.uw)} | {_r(a.ew, a.uw)} |')
    return out


HEAD = ['| 区分 | 出走 | 1着 | 市場見込み | 実数÷市場 | z | 3着内 | 市場見込み | 実数÷市場 | z | 1着 実数÷均等 | 1着 市場÷均等 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']


def chi2_sf(x, k):
    """カイ二乗の上側確率（Wilson–Hilferty 近似。scipy なしで足りる精度）。"""
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 1 - NormalDist().cdf(z)


def heterogeneity(groups):
    """コースごとの log(実数/見込み) がコース間で違うか（DerSimonian–Laird）。groups: [(実数, 見込み)]。"""
    ys, ws = [], []
    for o, e in groups:
        if e < 1:
            continue
        ys.append(math.log((o + 0.5) / (e + 0.5)))
        ws.append(e)  # Poisson 近似で var(log) ≈ 1/E
    if len(ys) < 2:
        return None
    m = sum(w * y for w, y in zip(ws, ys)) / sum(ws)
    q = sum(w * (y - m) ** 2 for w, y in zip(ws, ys))
    k = len(ys) - 1
    c = sum(ws) - sum(w * w for w in ws) / sum(ws)
    tau2 = max(0.0, (q - k) / c)
    return {'k': len(ys), 'Q': q, 'p': chi2_sf(q, k), 'tau2': tau2, 'pooled': math.exp(m)}


def shrunk(o, e, pooled_ratio, tau2):
    """Gamma–Poisson の縮小推定: コースの比を全体の比へ寄せる（τ²=0 なら全体の比そのもの）。"""
    a = 1 / tau2 if tau2 > 1e-9 else 1e9
    return (o + a * pooled_ratio) / (e + a)


def change_type(x):
    if x['prev_zone'] is None or x['prev_zone'] == x['zone']:
        return None
    if x['zone'] == '内':
        return 'turf_to_inner'
    if x['zone'] == '外':
        return 'turf_to_outer'
    return None  # 中への替わりは現行フラグなし


def loglik(races, adj):
    """1着馬の対数確率の合計。adj(race, row) は勝率に掛ける倍率。"""
    ll = 0.0
    for r in races:
        q = {x['num']: x['pw'] * adj(r, x) for x in r['rows']}
        s = sum(q.values())
        ll += math.log(next(q[x['num']] / s for x in r['rows'] if x['win']))
    return ll


def summary(key, courses, bonf, n_tests, het, exc):
    """冒頭の要点（数字は本文と同じ計算から）。"""
    po, ci_o, mo, pi, ci_i, mi = key['2021/10〜']
    po2, ci_o2, mo2, pi2, ci_i2, mi2 = key['2016/10〜2021/9']
    c = key.get('corr', (float('nan'), 0))
    k, n, e, tail = key['mcs']
    g = lambda pre: next((v for k_, v in key.items() if isinstance(k_, str) and k_.startswith(pre)), (float('nan'), float('nan')))
    h_out = het.get(('2021/10〜', '外', '1着 実数÷均等'))
    return [
        '## 要点', '',
        f'1. **全コース共通で見ると、外枠替わり −3 は実績より強い。** 芝重賞の外枠（7・8枠）の中枠に対する効果を点数に直すと、'
        f'2021/10〜 で {po:+.1f}点（95%区間 {ci_o[0]:+.1f}〜{ci_o[1]:+.1f}）、2016/10〜2021/9 で {po2:+.1f}点（{ci_o2[0]:+.1f}〜{ci_o2[1]:+.1f}）。'
        f'市場の割り引きは {mo:+.1f}点／{mo2:+.1f}点。−3 はどちらの期間でも95%区間の外。'
        f'内枠は {pi:+.1f}点（{ci_i[0]:+.1f}〜{ci_i[1]:+.1f}）／{pi2:+.1f}点（{ci_i2[0]:+.1f}〜{ci_i2[1]:+.1f}）で、+2 は区間内（市場は {mi:+.1f}／{mi2:+.1f}点）',
        f'2. **前走から内外が変わった馬は、市場の見込みどおりに走っている。** 市場にフラグの倍率（内 ×{math.exp(FLAGS["turf_to_inner"] / T):.2f}・外 ×{math.exp(FLAGS["turf_to_outer"] / T):.2f}）'
        f'を掛けると、評価年（2024〜）の1着馬の対数尤度は {g("案C ")[0]:+.1f}、2016/10〜2021/9 では {g("案C ")[1]:+.1f}（市場だけより悪い）。'
        f'turf_to_outer だけ外すと {g("（参考）turf_to_outer")[0]:+.1f}／{g("（参考）turf_to_outer")[1]:+.1f}、turf_to_inner だけ外すと '
        f'{g("（参考）turf_to_inner")[0]:+.1f}／{g("（参考）turf_to_inner")[1]:+.1f} なので、損の大半は turf_to_outer。'
        f'学習期間で最も当てはまる共通の点数（−4〜+4 を0.5刻み）は 内替わり {key["best"][0]:+g}・外替わり {key["best"][1]:+g}（§5）。替わり組の 実数÷市場 は 1着 1.07〜1.12・3着内 0.96〜1.06（§4）',
        f'3. **コースごとの違いは、今のデータ量では偶然と区別できない。** {len(courses)}コース（2021/10〜 に12レース以上）で、'
        f'外枠の比のコース間のばらつきは偶然の幅に収まり（1着 実数÷均等 τ²={h_out["tau2"]:.3f}, p={h_out["p"]:.2f}。3着内・市場比は τ²=0）、'
        f'学習期間と評価期間のコース別の外枠の比の相関は {c[0]:+.2f}（{c[1]}コース、続いていない）。Bonferroni の目安（|z|>{bonf:.2f}、{n_tests}通り）を超えたコースはない。'
        '1コースの外枠の勝ち数は2021/10〜 で1〜12勝なので、コース別の点数の95%区間は ±6〜±16点と広い',
        f'4. **マイルCS（京都芝1600外）**: データ内の京都開催 {n} 回で外枠 {k} 勝（市場の見込み {e:.1f} 勝、{k} 勝以上の確率 {tail:.3f}）。'
        'ただしレースと区分を見てから選んだ比較で、同コースの他の重賞を含めると外枠は市場どおり（§6）。'
        'このレースだけ turf_to_outer を外した判断は、コース固有の傾向というより「−3 が全体に強すぎる」ことの一例と読む方が数字に合う',
        '',
        '### 案の比較（採用はユーザー判断）', '',
        '| 案 | 中身 | 評価年 2024〜 の差（対数尤度、市場だけ比） | 2016/10〜2021/9 の差 | 長所 | 短所 |',
        '|---|---|---:|---:|---|---|',
        f'| A コース別の表で置き換え | コース×替わりの倍率を学習期間から縮小推定 | {g("案A ")[0]:+.1f} | {g("案A ")[1]:+.1f} | 各コースの傾向を入れる形にはなる | '
        'コース差が検出できないため、縮小すると全コースほぼ同じ値（≈0点）に潰れる。実質は「フラグを外す」と同じで、表を持つ手間と見かけの精密さだけが増える |',
        f'| B フラグを残してコース例外 | −3 を残し、外枠が走るコースだけ外す（事前基準: 学習期間の外枠3着内 z≥+1 → {"、".join(exc) or "なし"}） | '
        f'{g("案B ")[0]:+.1f} | {g("案B ")[1]:+.1f} | 今のマイルCSの運用に近く、変更が小さい | −3 自体が強すぎる問題が残り、例外コースを選ぶ基準が後付けになりやすい（多重比較） |',
        f'| B′ マイルCS型の例外だけ | 京都芝1600外だけ turf_to_outer なし | {g("案B′")[0]:+.1f} | {g("案B′")[1]:+.1f} | 今回の判断をそのまま固定 | 同上。効果はほぼ無い |',
        f'| C 現状維持 | 全コース共通 +2/−3 | {g("案C ")[0]:+.1f} | {g("案C ")[1]:+.1f} | 変更なし | 外枠替わりの馬を実績より割り引き続ける |',
        '',
        '数字の上では「共通のフラグのまま、turf_to_outer を小さくする（または外す）」が A・B より素直だが、これは案として並べるだけで、'
        '重み・ルールの変更はユーザーが決める。数レースの結果ではなく、約1,000レースの重賞全体で −3 が区間外という根拠なので、'
        '変えるならコース別より先に共通の値の見直しになる。どの案でも、engine では点数を ability に足してから市場と 25% で混ぜるので、'
        '§5 の対数尤度の差は「市場に上乗せした場合」の大きさで、engine の買い目への影響はこれより小さい。', '',
        '### 注意', '',
        '- 多重比較: コース×区分×指標で数十通りを見ている。|z|>2 は偶然でも数個出る。コース・区分を見てから選んだ比較（マイルCSの外枠など）は、'
        '表の z や確率より実際の確からしさは低い',
        '- 学習（2021/10〜2023）と評価（2024〜）は年で分け、例外・表・最良の点数は学習期間だけで決めた。2016/10〜2021/9 は独立の再現確認',
        '- 前走の枠は馬の戦績キャッシュから取る（2021/10〜 は98%、それ以前はキャッシュのある馬だけ）。前走が海外・地方でも枠番があれば使う',
        '- 枠は抽選なので「実数÷均等」は枠の効果の偏りの無い推定だが、能力の偏りの分だけ誤差は Poisson 近似より大きい。'
        '1着は勝ち数が少ないので、3着内（市場比）も併せて見る',
        '- コースは競馬場＋距離＋外/内回り。馬場状態・開催週（内の荒れ）・頭数は分けていない。京都は 2020/11〜2023/4 に休止',
        '- 対象は重賞だけ。条件戦を含めればコース別の数は増えるが、市場の効率や枠の効果が重賞と同じとは限らない', '']


def main(out=OUT):
    races = load()
    main_r = [r for r in races if r['date'] >= MAIN_FROM]
    counts = defaultdict(int)
    for r in main_r:
        counts[r['course']] += 1
    courses = sorted((c for c, k in counts.items() if k >= MIN_RACES), key=lambda c: -counts[c])
    has_prev = sum(x['prev_zone'] is not None for r in main_r for x in r['rows'])
    total = sum(len(r['rows']) for r in main_r)

    # 1. 全体とコース別の枠区分
    acc = defaultdict(lambda: defaultdict(Acc))  # (course|'全体', period) -> zone -> Acc
    for r in races:
        per = period(r)
        for x in r['rows']:
            for c in ('全体', r['course']):
                acc[(c, per)][x['zone']].add(x)
                if per != '2016/10〜2021/9':
                    acc[(c, '2021/10〜')][x['zone']].add(x)

    n_tests = len(courses) * len(ZONES) * 2
    bonf = NormalDist().inv_cdf(1 - 0.05 / 2 / n_tests)

    L = ['# 芝のコース別の枠（内・中・外）と枠替わりフラグの照合', '',
         '研究のみ。engine・RULES は変えていない。再実行: `python3 -m keiba.research.course_gate_check`', '',
         f'- 対象: JRA 平地芝重賞（keiba/data、{races[0]["date"]}〜{races[-1]["date"]}、頭数{engine.SMALL_FIELD + 1}頭以上の'
         f'{len(races)}レース、うち 2021/10〜 は {len(main_r)}レース）。コースは競馬場＋距離＋外/内回り',
         '- 区分: 1・2枠＝内、3〜6枠＝中、7・8枠＝外（engine.gate_zone_of）',
         '- 市場の見込み: 勝率＝1/単勝オッズの正規化、3着内＝単勝市場の λ で補正した Benter 型（form_check と同じ）。'
         'z＝(実数−見込み)/√Σp(1−p)',
         '- 均等: 1着＝1/頭数。枠は抽選なので **1着 実数÷均等** は枠そのものの効果（能力の偏りは誤差）、'
         '**市場÷均等** は市場が枠をどれだけ織り込んでいるか、**実数÷市場** は市場の織り込み残し',
         f'- 期間: 学習＝2021/10〜2023年、評価＝2024年〜（年で分割）。2016/10〜2021/9 は独立の再現確認。'
         f'京都は 2020/11〜2023/4 改修で休止し、その間の京都の重賞は阪神・中京などで行われた（コースは実際の開催場で数える）',
         f'- 前走の枠: 2021/10〜 の出走 {total} 頭のうち {has_prev} 頭（{has_prev / total:.0%}）で取れた（馬の戦績キャッシュ）。'
         f'現行の点数: turf_to_inner {FLAGS.get("turf_to_inner"):+}、turf_to_outer {FLAGS.get("turf_to_outer"):+}、温度 {T:g}'
         f'（勝率の倍率 内替わり ×{math.exp(FLAGS["turf_to_inner"] / T):.2f}、外替わり ×{math.exp(FLAGS["turf_to_outer"] / T):.2f}。'
         '他馬との比で、正規化前）', '']
    hdr = len(L)
    key = {}

    L += ['## 1. 芝重賞全体の枠区分', ''] + HEAD
    for per in ('2016/10〜2021/9', '学習（2021/10〜2023）', '評価（2024〜）', '2021/10〜'):
        L += zone_rows(acc[('全体', per)], f'{per} ')
    a = acc[('全体', '2021/10〜')]
    L += ['', f"2021/10〜 の外枠の1着 実数÷均等 {a['外'].w / a['外'].uw:.2f}、中 {a['中'].w / a['中'].uw:.2f}、"
              f"内 {a['内'].w / a['内'].uw:.2f}（外÷中 {a['外'].w / a['外'].uw / (a['中'].w / a['中'].uw):.2f}）。", '']

    # 2. コース別
    L += ['## 2. コース別（2021/10〜 に {} レース以上）'.format(MIN_RACES), '',
          f'{len(courses)}コース × 3区分 × 2（1着・3着内）＝{n_tests} 通りを見ているので、偶然だけでも |z|>2 が {n_tests * 0.0455:.1f} 個ほど出る。'
          f'Bonferroni の目安は |z|>{bonf:.2f}。', '']
    het_rows = []
    for c in courses:
        L += [f'### {c}（2021/10〜 {counts[c]}レース）', ''] + HEAD
        for per in ('学習（2021/10〜2023）', '評価（2024〜）', '2021/10〜', '2016/10〜2021/9'):
            L += zone_rows(acc[(c, per)], f'{per} ')
        L.append('')
    # 一覧
    L += ['### コース別の一覧（2021/10〜、外枠）', '',
          '| コース | レース | 外 出走 | 外 1着 | 外 1着 実数÷市場 | z | 外 3着内 実数÷市場 | z | 外 1着 実数÷均等 | 外÷中（1着 実数÷均等） | '
          '学習 外 3着内 実数÷市場 | 評価 外 3着内 実数÷市場 | 2016〜21 外 3着内 実数÷市場 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    flagged = []
    for c in courses:
        o = acc[(c, '2021/10〜')]['外']
        m = acc[(c, '2021/10〜')]['中']
        ratio_om = (o.w / o.uw) / (m.w / m.uw) if m.w and m.uw and o.uw else float('nan')
        lt = acc[(c, '学習（2021/10〜2023）')]['外']
        ev = acc[(c, '評価（2024〜）')]['外']
        old = acc[(c, '2016/10〜2021/9')]['外']
        L.append(f'| {c} | {counts[c]} | {o.n} | {o.w} | {_r(o.w, o.ew)} | {o.zw():+.1f} | {_r(o.t, o.et)} | {o.zt():+.1f} | '
                 f'{_r(o.w, o.uw)} | {ratio_om:.2f} | {_r(lt.t, lt.et)} | {_r(ev.t, ev.et)} | {_r(old.t, old.et)} |')
        if abs(o.zw()) > bonf or abs(o.zt()) > bonf:
            flagged.append(c)
    L += ['', f'Bonferroni の目安を超えた外枠: {"、".join(flagged) if flagged else "なし"}。', '']

    # 点数に直した枠の効果（外÷中、内÷中。1着 実数÷均等 の比を 温度×log にする）
    L += ['### 点数に直した枠そのものの効果（温度8、1着 実数÷均等 の比）', '',
          '外÷中・内÷中 の比を 点＝温度×log(比) にした。95%区間は勝ち数の Poisson 近似（√(1/勝ち数₁+1/勝ち数₂)、能力の偏りの分だけ実際はやや広い）。'
          f'現行は「中→外」で {FLAGS["turf_to_outer"]:+}、「中→内」で {FLAGS["turf_to_inner"]:+}。'
          '「市場」は同じ比を市場の見込み（市場÷均等）で出したもの（市場が枠をどれだけ割り引いているか）。', '',
          '| コース | 期間 | 外 1着 | 中 1着 | 外÷中 の点（実数） | 95%区間 | 市場 | 内 1着 | 内÷中 の点（実数） | 95%区間 | 市場 | 外の −3 は区間外か |',
          '|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|']

    def pts(a, b):
        if not a.w or not b.w:
            return None, None, None
        p = T * math.log((a.w / a.uw) / (b.w / b.uw))
        se = T * math.sqrt(1 / a.w + 1 / b.w)
        pm = T * math.log((a.ew / a.uw) / (b.ew / b.uw))
        return p, (p - 1.96 * se, p + 1.96 * se), pm

    out_of = []
    for c, per in [('全体', '2021/10〜'), ('全体', '2016/10〜2021/9')] + [(c, '2021/10〜') for c in courses]:
        g = acc[(c, per)]
        po, ci_o, mo = pts(g['外'], g['中'])
        pi, ci_i, mi = pts(g['内'], g['中'])
        f = lambda v: '—' if v is None else f'{v:+.1f}'
        fc = lambda ci: '—' if ci is None else f'{ci[0]:+.1f}〜{ci[1]:+.1f}'
        mark = '—' if ci_o is None else ('区間より下（−3 でも足りない）' if ci_o[1] < FLAGS['turf_to_outer'] else
                                         '区間外（−3 は強すぎ）' if ci_o[0] > FLAGS['turf_to_outer'] else '区間内')
        if c == '全体':
            key[per] = (po, ci_o, mo, pi, ci_i, mi)
        if c != '全体' and ci_o is not None and ci_o[0] > FLAGS['turf_to_outer']:
            out_of.append(c)
        L.append(f"| {c} | {per} | {g['外'].w} | {g['中'].w} | {f(po)} | {fc(ci_o)} | {f(mo)} | {g['内'].w} | {f(pi)} | {fc(ci_i)} | "
                 f"{f(mi)} | {mark} |")
    L += ['', f'コース別で −3 が95%区間の外（強すぎ）のコース: {"、".join(out_of) if out_of else "なし"}'
              f'（{len(courses)}コースを見て選んでいるので、個々の区間は多重比較の分だけ甘い）。', '']

    # 3. コース間の違いがあるか（異質性）
    L += ['## 3. コース間で枠の効果が違うか（異質性の検定）', '',
          'コースごとの log(実数/見込み) のばらつきが、偶然（Poisson）で説明できる幅を超えるか。Q と p（自由度＝コース数−1）、'
          'τ²＝偶然を超えるばらつき（0 ならコース差は検出できない）。対象はコース別の表と同じコース（2021/10〜）と、'
          '2016/10〜2021/9 で同じ基準（12レース以上）を満たすコース。', '',
          '| 期間 | 区分 | 指標 | コース数 | Q | p | τ² | 全体の比 |', '|---|---|---|---:|---:|---:|---:|---:|']
    old_counts = defaultdict(int)
    for r in races:
        if r['date'] < MAIN_FROM:
            old_counts[r['course']] += 1
    old_courses = [c for c, k in old_counts.items() if k >= MIN_RACES]
    het = {}
    for per, cs in (('2021/10〜', courses), ('2016/10〜2021/9', old_courses)):
        for z in ZONES:
            for lab, base in (('3着内 実数÷市場', lambda a: (a.t, a.et)), ('1着 実数÷市場', lambda a: (a.w, a.ew)),
                              ('1着 実数÷均等', lambda a: (a.w, a.uw))):
                h = heterogeneity([base(acc[(c, per)][z]) for c in cs])
                het[(per, z, lab)] = h
                if h:
                    L.append(f"| {per} | {z} | {lab} | {h['k']} | {h['Q']:.1f} | {h['p']:.2f} | {h['tau2']:.3f} | {h['pooled']:.2f} |")
    L.append('')

    # 再現性: 学習期間のコース別の外枠の比が評価期間の比を当てるか
    pairs = []
    for c in courses:
        lt = acc[(c, '学習（2021/10〜2023）')]['外']
        ev = acc[(c, '評価（2024〜）')]['外']
        if lt.et >= 3 and ev.et >= 3:
            pairs.append((c, math.log((lt.t + .5) / (lt.et + .5)), math.log((ev.t + .5) / (ev.et + .5))))
    if len(pairs) >= 3:
        xs, ys = [p[1] for p in pairs], [p[2] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        sy = math.sqrt(sum((y - my) ** 2 for y in ys))
        corr = sxy / sx / sy if sx and sy else float('nan')
        key['corr'] = (corr, len(pairs))
        L += [f'学習（2021/10〜2023）と評価（2024〜）のコース別の外枠 3着内 log(実数÷市場) の相関: {corr:+.2f}（{len(pairs)}コース）。'
              '正なら、学習期間の「このコースは外枠が市場より走る/走らない」が評価期間でも続いた、という意味。', '']

    # 4. 枠替わり
    ch = defaultdict(lambda: defaultdict(Acc))  # period -> (prev, now) -> Acc
    for r in races:
        if r['date'] < MAIN_FROM:
            continue
        per = period(r)
        for x in r['rows']:
            if x['prev_zone'] is None:
                continue
            for p in (per, '2021/10〜'):
                ch[p][(x['prev_zone'], x['zone'])].add(x)
    L += ['## 4. 前走から内外が変わった馬（2021/10〜）', '',
          '| 期間 | 前走→今回 | 現行フラグ | 出走 | 1着 | 市場見込み | 実数÷市場 | z | 3着内 | 実数÷市場 | z | 1着 実数÷均等 | 1着 市場÷均等 |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for per in ('学習（2021/10〜2023）', '評価（2024〜）', '2021/10〜'):
        for pz in ZONES:
            for nz in ZONES:
                a = ch[per].get((pz, nz))
                if not a or not a.n:
                    continue
                f = 'turf_to_inner +2' if nz == '内' and pz != '内' else 'turf_to_outer −3' if nz == '外' and pz != '外' else ''
                L.append(f'| {per} | {pz}→{nz} | {f} | {a.n} | {a.w} | {a.ew:.1f} | {_r(a.w, a.ew)} | {a.zw():+.1f} | {a.t} | '
                         f'{_r(a.t, a.et)} | {a.zt():+.1f} | {_r(a.w, a.uw)} | {_r(a.ew, a.uw)} |')
    L.append('')

    # 枠そのものの効果（実数÷均等）で見た、替わり組と「今回と同じ区分に前走もいた組」の比
    a = ch['2021/10〜']

    def eff(keys, attr='w', base='uw'):
        o = sum(getattr(a[k], attr) for k in keys if k in a)
        e = sum(getattr(a[k], base) for k in keys if k in a)
        return o / e if e else float('nan')

    to_out = eff([('内', '外'), ('中', '外')])
    stay_mid = eff([('中', '中')])
    to_in = eff([('中', '内'), ('外', '内')])
    m_to_out = eff([('内', '外'), ('中', '外')], 'ew')
    m_stay_mid = eff([('中', '中')], 'ew')
    m_to_in = eff([('中', '内'), ('外', '内')], 'ew')
    L += ['### 現行の点数との比較', '',
          '| 組 | 現行の倍率（温度8） | 実数で見た倍率（1着 実数÷均等 の比） | 市場の倍率（市場÷均等 の比） |', '|---|---:|---:|---:|',
          f'| 外替わり ÷ 中→中 | ×{math.exp(FLAGS["turf_to_outer"] / T):.2f} | ×{to_out / stay_mid:.2f} | ×{m_to_out / m_stay_mid:.2f} |',
          f'| 内替わり ÷ 中→中 | ×{math.exp(FLAGS["turf_to_inner"] / T):.2f} | ×{to_in / stay_mid:.2f} | ×{m_to_in / m_stay_mid:.2f} |', '',
          '実数で見た倍率は勝ち数が少なく誤差が大きい（下の対数尤度の方が直接の検証）。', '']

    # 5. 評価年での対数尤度: 市場に倍率を掛けて当てはまりが良くなるか
    learn = [r for r in races if MAIN_FROM <= r['date'] < EVAL_FROM]
    evalr = [r for r in races if r['date'] >= EVAL_FROM]
    old = [r for r in races if r['date'] < MAIN_FROM]

    def flag_mult(r, x, table=None, exceptions=()):
        t = change_type(x)
        if not t:
            return 1.0
        if table is not None:
            return table.get((r['course'], t), table.get(('*', t), 1.0))
        if t == 'turf_to_outer' and r['course'] in exceptions:
            return 1.0
        return math.exp(FLAGS[t] / T)

    # 学習: 替わり組の 1着 実数÷市場（全体）と、コース別の縮小推定
    def learn_table(rs):
        g = defaultdict(lambda: [0, 0.0])
        for r in rs:
            for x in r['rows']:
                t = change_type(x)
                if t:
                    for key in (('*', t), (r['course'], t)):
                        g[key][0] += x['win']
                        g[key][1] += x['pw']
        table = {}
        info = {}
        for t in ('turf_to_inner', 'turf_to_outer'):
            o, e = g[('*', t)]
            pooled = o / e if e else 1.0
            h = heterogeneity([tuple(v) for k, v in g.items() if k[0] != '*' and k[1] == t])
            tau2 = h['tau2'] if h else 0.0
            table[('*', t)] = pooled
            for k, (oc, ec) in g.items():
                if k[0] != '*' and k[1] == t:
                    table[k] = shrunk(oc, ec, pooled, tau2)
            info[t] = (o, e, pooled, tau2, h)
        return table, info

    table, info = learn_table(learn)
    # 例外: 学習期間の外枠（全馬）の 3着内 z が +1 以上のコースは turf_to_outer を付けない（事前に決めた基準）
    exc = sorted(c for c in counts if acc[(c, '学習（2021/10〜2023）')]['外'].et >= 3 and
                 acc[(c, '学習（2021/10〜2023）')]['外'].zt() >= 1.0)
    # 全馬の今回の区分でコース別に掛ける表（替わりでなく今回の枠そのもの）。学習期間の 1着 実数÷市場 を縮小
    zt = {}
    for z in ZONES:
        g = [(c, acc[(c, '学習（2021/10〜2023）')][z]) for c in counts]
        h = heterogeneity([(a_.w, a_.ew) for _, a_ in g])
        tot = acc[('全体', '学習（2021/10〜2023）')][z]
        pooled = tot.w / tot.ew
        for c, a_ in g:
            zt[(c, z)] = shrunk(a_.w, a_.ew, pooled, h['tau2'] if h else 0.0)
        zt[('*', z)] = pooled

    # 学習期間で最も当てはまる共通の点数（市場に上乗せ）を格子で探す
    grid = [k / 2 for k in range(-8, 9)]

    def fixed(pi, po):
        def f(r, x):
            t = change_type(x)
            return 1.0 if not t else math.exp((pi if t == 'turf_to_inner' else po) / T)
        return f

    best = max(((pi, po) for pi in grid for po in grid), key=lambda pp: loglik(learn, fixed(*pp)))
    key['best'] = best

    base = loglik(evalr, lambda r, x: 1.0)
    variants = [
        ('市場だけ（＝枠替わりフラグなし）', lambda r, x: 1.0),
        ('案C 現状維持（全コース共通の +2/−3）', lambda r, x: flag_mult(r, x)),
        (f'案B フラグを残してコース例外（学習期間に外枠の3着内 z≥+1 のコースは turf_to_outer なし: {"、".join(exc) or "なし"}）',
         lambda r, x: flag_mult(r, x, exceptions=exc)),
        ('案B′ マイルCS型の例外だけ（京都芝1600外で turf_to_outer なし）', lambda r, x: flag_mult(r, x, exceptions=('京都芝1600外',))),
        ('案A コース別の表で置き換え（替わり組の倍率を学習期間からコース別に縮小推定）', lambda r, x: flag_mult(r, x, table=table)),
        ('（参考）turf_to_outer だけ外す（+2/0）', fixed(FLAGS['turf_to_inner'], 0)),
        ('（参考）turf_to_inner だけ外す（0/−3）', fixed(0, FLAGS['turf_to_outer'])),
        ('（参考）現行の半分（+1/−1.5）', fixed(FLAGS['turf_to_inner'] / 2, FLAGS['turf_to_outer'] / 2)),
        (f'（参考）学習期間で最も当てはまる共通の点数（内替わり {best[0]:+g}・外替わり {best[1]:+g}）', fixed(*best)),
        ('（参考）今回の区分そのものにコース別の倍率（学習期間の1着 実数÷市場を縮小）',
         lambda r, x: zt.get((r['course'], x['zone']), zt[('*', x['zone'])])),
    ]
    L += ['## 5. 案の比較（評価年 2024〜 の1着馬の対数尤度）', '',
          f'市場の勝率に各案の倍率を掛けて正規化し、評価期間 {len(evalr)} レースの1着馬の対数確率を足した。差が正なら市場だけより当たる。'
          '倍率・例外・表は学習期間（2021/10〜2023）だけで決めた。参考に 2016/10〜2021/9（前走の枠が取れた馬だけ替わり扱い）でも同じ計算をした。'
          'engine では点数は ability に足してから市場と混ぜる（MODEL_WEIGHT）ので、ここは「市場が織り込んでいない分があるか」の検証で、'
          '枠そのものの効果の大きさとは別（§4の表）。', '',
          '| 案 | 評価 2024〜 の対数尤度 | 市場だけとの差 | 2016/10〜2021/9 の差 |', '|---|---:|---:|---:|']
    base_old = loglik(old, lambda r, x: 1.0)
    for lab, f in variants:
        ll = loglik(evalr, f)
        llo = loglik(old, f)
        key[lab] = (ll - base, llo - base_old)
        L.append(f'| {lab} | {ll:.2f} | {ll - base:+.2f} | {llo - base_old:+.2f} |')
    L.append('')
    for t in ('turf_to_inner', 'turf_to_outer'):
        o, e, pooled, tau2, h = info[t]
        L.append(f'- 学習期間の {t} の馬: 1着 {o} / 市場見込み {e:.1f}（実数÷市場 {pooled:.2f}。現行の倍率は ×{math.exp(FLAGS[t] / T):.2f}）。'
                 f'コース間の τ²={tau2:.3f}' + (f'（Q={h["Q"]:.1f}, p={h["p"]:.2f}, {h["k"]}コース）' if h else ''))
    L.append('')

    # 6. マイルCS（京都芝1600外）
    mcs = [r for r in races if r['course'] == '京都芝1600外' and 'マイルチャンピオン' in r['name']]
    k16 = defaultdict(Acc)
    for r in races:
        if r['course'] == '京都芝1600外':
            for x in r['rows']:
                k16[x['zone']].add(x)
    L += ['## 6. 京都芝1600外回り（マイルCSのコース）', '',
          f"データ内のマイルCS（京都開催）: {', '.join(r['date'][:4] for r in mcs)}（2020〜2022 は阪神）。"
          f"京都芝1600外の重賞は全期間で {sum(1 for r in races if r['course'] == '京都芝1600外')} レース。", ''] + HEAD + zone_rows(k16, '全期間 ')
    wins = defaultdict(int)
    for r in mcs:
        for x in r['rows']:
            if x['win']:
                wins[x['zone']] += 1
    # 外枠が勝つ確率（各回の市場の外枠の勝率の和）から、5勝以上の確率（各回独立のポアソン二項）
    probs = [sum(x['pw'] for x in r['rows'] if x['zone'] == '外') for r in mcs]
    dist = [1.0]
    for q in probs:
        dist = [a * (1 - q) + (dist[i - 1] * q if i else 0) for i, a in enumerate(dist + [0.0])]
    k = wins.get('外', 0)
    tail = sum(dist[k:])
    key['mcs'] = (k, len(mcs), sum(probs), tail)
    L += ['', f"データ内のマイルCS（京都）の勝ち馬の区分: {dict(wins)}。市場の見込みでは外枠の勝ちは {len(mcs)} 回で計 {sum(probs):.1f} 勝、"
              f"{k} 勝以上になる確率は {tail:.3f}。ただし「マイルCS × 外枠」はレースと区分を見てから選んだ比較なので、"
              f"芝重賞の数十のレース × 3区分のどこかでこの程度の偏りが出るのは珍しくない。同じコースの他の重賞を含めると外枠は"
              f"{k16['外'].w} 勝／市場見込み {k16['外'].ew:.1f}。", '']
    L[hdr:hdr] = summary(key, courses, bonf, n_tests, het, exc)
    Path(out).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main(*sys.argv[1:])
