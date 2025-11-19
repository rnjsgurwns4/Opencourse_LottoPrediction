# Kotlin + Weka ML 로또 예측기

> 낯선 도구 해커톤(Kotlin, Ktor, Weka)과 고난도 문제(ML 로또 예측)를 같이 사용한 해커톤 미션

과거 로또 당첨 데이터로 머신러닝 모델을 훈련시켜, 예측이 가능한가지에 대한 문제를 해결하기 위한 프로젝트

Kotlin/Ktor 기반의 간단한 웹 사이트를 통해 3가지 ML 모델의 성능을 실시간으로 비교하고, 미래의 로또 번호를 예측

## 1. 핵심 기능 및 실행 화면

### 1) 메인 페이지

Ktor 웹 서버 `http://localhost:8080`에서 실행되며, 두 가지 핵심 기능으로 연결됨

* **[미래] 예측하기:** 훈련된 모델을 사용해 다음 회차 번호를 예측
* **[과거] 검증하기:** 가장 최근 회차를 정답지로 두고, 3개 모델의 성능을 비교한 리포트 확인
* **[통계] 보기:** 분석한 데이터들을 차트로 시각화

<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/866a29ed-f8cb-4f2a-b23a-d97e552c4cb1" />



### 2) 과거 검증


"과거 검증하기" 버튼 클릭 시, 가장 최근 회차를 기준으로 3개 모델이 3세트씩 제출한 예측 결과와 실제 등수를 비교

* `RandomForest`, `Logistic`, `J48` 모델이 경쟁
* 각 모델은 `[Top 6 1세트] + [확률 샘플링 2세트]`를 제출
* 이 결과를 바탕으로 2개의 모델을 선발

<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/2469d53a-0314-410e-92dc-fc338f393145" />


### 3) 미래 예측

"미래 예측하기"를 누르면, 2가지 전략 중 하나를 선택하는 페이지로 이동

1.  **최고 등수 우선:** 4등 1개 > 5등 3개 (가장 높은 등수)
2.  **총 당첨 횟수 우선:** 5등 3개 > 4등 1개 (총 맞춘 개수)

<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/5de9f09e-2e19-47c1-9cf6-3470e1fa6158" />
<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/983b5ae3-31c6-45e5-aff0-b785491bd518" />


전략과 생성할 세트 개수(n)를 선택하면, 해당 전략의 모델이 로또 번호 n세트를 생성

<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/b86678a1-bee8-4fed-bb0d-81af454ff876" />


### 4) 과거 데이터 시각화

분석에 사용된 데이터들을 시각화

<img width="1280" height="764" alt="image" src="https://github.com/user-attachments/assets/27c301cf-6ae9-4494-9dd7-f13710dd7785" />


### 5) 실행 영상 링크

추가 예정

---

## 2. 기술 스택

* **Language:** Kotlin (JVM) 1.9+
* **Web Server:** Ktor
* **Data Handling:** kotlinx-dataframe
* **ML Engine:** Weka (Logistic, RandomForest, J48)
* **Data Fetching:** Ktor Client, kotlinx-serialization
* **Build:** Gradle (KTS)

---

## 3. 프로젝트 구조

`src/main/kotlin/lotto/` 패키지 내부의 주요 파일 및 역할

### 핵심 로직
* **TrainingService.kt**: 전체 파이프라인을 총괄하는 서비스입니다.
    * performTotalTraining(): 데이터 수집 → 과거 검증 → 모델 선발 → 미래 예측기 훈련 → 전역 상태 업데이트의 전 과정을 수행
    * trainAllModels(): 정의된 모든 모델(Logistic, RF, J48 등)을 일괄 훈련
* **LottoModelTrainer.kt**: Weka 라이브러리와 직접 통신하여 모델을 학습
    * train(): 전체 데이터를 45개(번호별) 그룹으로 나누고, AbstractClassifier.makeCopy를 사용해 45개의 개별 모델을 복제/훈련
* **FeatureEngineer.kt**: 원본 데이터를 ML 학습용 특성(Feature)으로 변환
    * createTrainingData(): 과거 이력을 바탕으로 5가지 특성(Recency, Freq_Short, Freq_Mid, Total_Main, Total_Bonus)을 계산하여 DataFrame을 생성
        * **Recency (미출현 기간):** 해당 번호가 마지막으로 당첨된 후 경과한 회차 수
        * **Freq_Short (단기 빈도):** 최근 10회차 내 출현 횟수 (단기 추세)
        * **Freq_Mid (중기 빈도):** 최근 25회차 내 출현 횟수 (중기 추세)
        * **Total_Main (누적 빈도):** 1회차부터 현재까지 메인 번호로 당첨된 총 횟수
        * **Total_Bonus (보너스 빈도):** 1회차부터 현재까지 보너스 번호로 나온 총 횟수
* **LottoPredictor.kt**: 훈련된 모델을 사용해 확률을 계산하고 번호를 생성
    * predictNextDraw(): 45개 모델의 확률값을 취합
    * getWeightedRandomSample(): 확률 기반 가중치 랜덤 샘플링(Roulette Wheel Selection) 알고리즘을 구현

### 웹 & 유틸리티
* **Main.kt**: 애플리케이션의 진입점
    * 초기 훈련을 트리거하고, Ktor 웹 서버를 실행하며 엔드포인트들을 정의
* **LottoDataManager.kt**: 동행복권 API 크롤링
    * fetchAllHistory(): 1회차부터 최신 회차까지의 데이터를 수집하고 파싱 오류(Missing Fields)를 처리
* **Scheduler.kt**: 자동화 시스템
    * 매주 토요일 저녁, 새로운 당첨 번호가 나오면 자동으로 데이터를 갱신하고 모델을 재학습
* **AppState.kt**: 훈련된 모델, 캐시된 데이터, 챔피언 정보 등 애플리케이션의 전역 상태(Singleton)를 관리
* **HtmlTemplates.kt**: CSS 스타일 및 결과 리포트 HTML 생성 로직

---

## 4. 실행 방법

### 1) 실행

1.  (터미널) Gradle을 통해 실행
    ```bash
    ./gradlew run
    ```

2.  프로그램이 실행되면, 콘솔에 모델 훈련 로그가 약 1~2분간 출력
    ```
    ...
    Ktor 웹 서버를 http://localhost:8080 에서 시작합니다.
    ```

3.  위 로그가 출력되면, 웹 브라우저에서 `http://localhost:8080`에 접속

---

## 5. 해결 과정

1.  1개 모델(Logistic), 1개 특성(freq_total: 1회차부터 당시까지 해당 번호가 누적 몇 번 나왔는지). Top 6 예측.
2.  3개 특성(recency: 최근 몇 주간 안 나왔는지, freq_latest: 최근 10주간 몇 번 나왔는지, freq_total) 추가.
3.  3개 모델(RF, J48) 경쟁 도입. 'Top 6' + '확률 기반 가중치 샘플링' 도입.
4.  2가지 '챔피언 선발 전략' (최고 등수 vs 최다 당첨) 웹 UI 추가. '보너스 번호', '중기(25주)' 특성을 추가하여 총 5개 특성으로 모델 성능 향상.
5. 기존 모델을 튜닝시켜 새롭게 추가된 모델을 포함한 총 5개의 모델로 로또 당첨 확률을 최대한 높이고자 함.


## 6. 결론

### 1) 정답이 없는 문제에 대한 해답

이 프로젝트는 "머신러닝으로 로또를 예측할 수 있나"라는 정답이 없는 문제로 시작.

5단계에 걸친 모델 고도화, 5가지 특성 공학, 하이퍼파라미터 튜닝을 적용한 결과, 현재 본인이 만든 모델로는 통계적으로 유의미한 예측이 불가능하다*는 결론에 도달.

5개의 모델이 3세트씩 제출해도 꽝(NONE)이 대부분이었으며 4, 5등조차 달성하기 어려움. 이것은 로또가 독립 시행이라는 통계학적 사실을 코드를 통해 과학적으로 증명하는 과정으로도 보임.

### 2) 낯선 도구에 대한 회고

이 미션의 또 다른 목표는 `Kotlin`, `Ktor`, `Weka`라는 '낯선 도구'에 익숙해지는 것.

`build.gradle` 의존성 문제, `kotlinx-dataframe`의 `Column not found` 스키마 인식 버그, `Weka`의 `AbstractClassifier.makeCopy` 참조 문제 등 수많은 기술적 난관이 있었음.

이 문제들을 다양한 로직을 통해 해결해 나가면서, 낯선 라이브러리의 내부 동작 원리와 코틀린 생태계를 이해할 수 있었음.

### 3) 향후 계획

AWS + Docker을 이용한 배포, 다양한 모델 추가 및 파라미터 튜닝을 통한 성능 향상.

### 4) 최종 요약

비록 로또 1등에 당첨되는 모델을 만들지는 못했지만, '정답이 없는 문제'에 대해 가설을 세우고, 낯선 도구를 사용해 시스템을 구축(Ktor 웹 앱)하며,

그 한계를 검증(과거 검증)하는 전 과정을 완수했다는 점에서 의미가 있다고 생각함.

거기다가 백그라운드 스케줄러와 웹 UI를 갖춘 서비스를 구축했다는 점에서 미션을 잘 완수했다고 봄.
