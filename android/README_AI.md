# HiCardi Monitor (Android) — AI 부정맥 분류

파이썬 데스크톱 앱(`hicardi_json.py`)을 안드로이드 모바일 앱(Kotlin)으로 옮긴 버전.
HiCardi JSON 심전도 파일을 열어 비트별로 재생하면서, **온디바이스 AI 모델**로 7클래스 부정맥을 분류한다.

## 구성
- 언어: Kotlin / UI: Android View + XML 레이아웃
- AI 런타임: **ExecuTorch** (`org.pytorch:executorch-android` 0.7.0, `app/libs/executorch.aar`)
- 모델: `app/src/main/assets/model.pte` (ExecuTorch, XNNPACK delegate)
  - 입력: float32 `[1, 1, 501]` — JSON `waveform`(z-score 정규화 상태) 그대로
  - 출력: float32 `[1, 7]` raw logits → 코드에서 `sigmoid` → 확률
- 7클래스: Normal / Sinus_Tachy / APC / AF_AFL / Bradycardia / VPC / Trigeminy

## 핵심 파일
- `MainActivity.kt` — JSON 파싱, 재생, HR/리듬 판정, 누적 소견, AI 결과 표시
- `EcgClassifier.kt` — `.pte` 로드 + 비트 추론(sigmoid)
- `EcgView.kt` — ECG 파형 커스텀 뷰

## 빌드 / 실행
```
cd hicardimonitor
./gradlew :app:assembleDebug      # APK: app/build/outputs/apk/debug/app-debug.apk
```
- Android Studio에서 `hicardimonitor` 폴더 Open 후 ▶ Run 해도 됨
- 지원 ABI: arm64-v8a(실폰), x86_64(에뮬레이터) — ExecuTorch 제약
- minSdk 26

## 검증됨 (x86_64 에뮬레이터, android-37)
- `model.pte` 로드 성공(`MODEL_LOAD_OK`)
- 온디바이스 추론 실행 성공(self-test + 비트 추론)
- 파이썬에서 동일 `.pte`로 낸 결과와 일치 (예: DN041 beat500 → AF_AFL ~0.95)

## 알려진 경고 (비치명적)
- "LOAD segment not aligned / 16 KB" — ExecuTorch 0.7.0 prebuilt `.so`가 16KB 페이지 정렬이
  아니라 Android 15+에서 호환 모드로 실행된다는 **경고**일 뿐, 동작에는 문제 없음.

## 참고
- 원본 `model.pte`는 ExecuTorch 프로그램이라, 파이썬 데스크톱 앱은 사실 이 모델을 로드하지
  못하고 label 모드로만 동작했음. 안드로이드 버전이 실제 모델 추론을 수행하는 최초 버전.
