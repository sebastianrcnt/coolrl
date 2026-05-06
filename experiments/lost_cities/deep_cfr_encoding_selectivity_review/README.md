# Lost Cities Deep CFR Encoding Selectivity Review

이 폴더는 Lost Cities Deep CFR input encoding이 expedition opening selectivity를 학습하기에 충분한지 검토한 코드 리뷰 record다.

## 배경

`deep_cfr_pure_self_play_zero_pit_poc_full_depth` 실험은 truncation bias를 크게 완화했고 recovery skill을 self-play로 학습했지만, safe 계열 상대에서 거의 전색을 여는 over-opening pattern은 유지했다.

이 record는 그 selectivity gap이 representation 문제와 관련 있는지 보기 위해 `src/coolrl/lost_cities/deep_cfr/encoding.py`를 검토한다.

## 파일

- [report.md](report.md): encoding 구조와 selectivity 관점의 발견
