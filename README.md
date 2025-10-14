# 20251013

## temp.csv 분석 가이드

`temp.csv`에는 삼성전자(005930.KS), 애플(AAPL), 엔비디아(NVDA)의 일별 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터가 동일한 파일 안에 저장되어 있습니다. 헤더가 두 줄(지표, 종목)로 구성되어 있기 때문에 일반적인 스프레드시트/통계 도구로 바로 분석하기가 번거롭습니다.

이 저장소에는 CSV를 정리하고 요약 통계를 출력해 주는 `analyze_temp_csv.py` 스크립트가 포함되어 있습니다. 표준 라이브러리만 사용하므로 추가 설치 없이 실행할 수 있습니다. 또한 `efficient_frontier.py`를 통해 동일한 파싱 로직을 재사용하여 일별 종가 수익률 기반의 효율적 투자선(efficient frontier)을 근사 계산할 수 있습니다.

```bash
python analyze_temp_csv.py                    # temp.csv 파일을 분석
python analyze_temp_csv.py 다른파일.csv          # 다른 파일을 분석하고 싶을 때
python efficient_frontier.py                  # temp.csv 기반 효율적 투자선 계산 + 그래프 저장
python efficient_frontier.py --no-plot        # 표 출력만 원할 때 (matplotlib 미설치 환경)
python efficient_frontier.py --plot-output my_frontier.png --step 0.02
```

스크립트는 다음과 같은 정보를 제공합니다.

- 데이터셋 개요: 거래 일수, 날짜 범위, 포함된 종목과 지표 목록
- 종목별 요약 통계: 각 지표의 최소/최대/평균값, 종가 기준 최고의/최악의 하루 수익률 및 해당 일자
- 종가 수익률 상관관계 행렬: 종목 간 수익률 동조화 정도 파악

`analyze_temp_csv.py`의 결과를 시작점으로 삼아 추가적인 시각화나 리포트를 만들고 싶다면, 동일한 파싱 로직을 재사용하거나 `describe_dataset` 함수를 직접 호출해서 문자열 대신 파이썬 객체 형태로 데이터를 가공할 수 있습니다. 효율적 투자선이 필요하다면 `efficient_frontier.py`가 일별 수익률을 정렬하고 공분산을 계산한 뒤, 지정된 격자 해상도(`--step`) 내에서 비지배 포트폴리오를 추출해 결과를 표 형태로 출력해 줍니다. 추가로 `matplotlib`가 설치되어 있다면 효율적 투자선의 산점도를 자동으로 저장해 시각 자료로 활용할 수 있습니다 (`--plot-output` 옵션으로 저장 경로 지정, `--no-plot`으로 비활성화 가능).
