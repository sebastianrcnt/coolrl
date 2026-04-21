# Omok Web

15×15 오목 쿨파고의 웹 버전. Svelte 5 + Vite + TypeScript 로 작성되어 있으며, 학습된 ONNX 모델을 브라우저에서 직접 평가(ONNX Runtime Web, WebGPU/WASM)하여 MCTS 로 수를 고른다.

## 요구 사항

- Node.js 20 이상 (Vite 8 요구)
- npm 10 이상

## 디렉토리 구조

```
src/
  App.svelte              # 루트 Svelte 컴포넌트 (DOM 바인딩만)
  main.ts                 # 엔트리 — App.svelte mount
  styles.css              # 전역 스타일
  core/                   # GameState, MCTS (DOM 비의존 순수 로직)
  evaluator/              # ONNX 워커 + 메인 스레드 클라이언트
  render/                 # Canvas 렌더러 (geometry / easing / stones / board)
  ui/                     # 애니메이션·상태 매니저 (StoneAnimations 등)
  util/                   # format / device / backend
  app/                    # OmokController (DomRefs 를 주입받는 조립 레이어)
public/
  best.onnx               # 기본 모델 (자동 로드)
index.html                # Svelte 앱 셸
```

## 설치

```bash
npm install
```

## 개발 서버

```bash
npm run dev
```

- `0.0.0.0` 바인딩이라 같은 네트워크의 모바일 기기에서도 `http://<PC_IP>:5173` 로 접속해 iOS/안드로이드 스모크 테스트 가능.
- `public/best.onnx` 가 있으면 첫 로드 시 자동으로 읽어 기본 모델로 쓴다.

## 프로덕션 빌드

```bash
npm run build        # dist/ 생성
npm run preview      # dist/ 를 로컬에서 서빙
```

Vite 가 워커를 별도 청크(`dist/assets/worker-*.js`)로 뽑아낸다. ONNX Runtime Web 은 번들에 포함되지 않고 워커 안에서 CDN(`importScripts`) 으로 로드된다.

## 타입 체크

```bash
npm run typecheck    # tsc --noEmit (strict)
```

Svelte 파일 내부 타입은 Vite + `@sveltejs/vite-plugin-svelte` 가 빌드 시 함께 검사한다.

## 테스트

```bash
npm test             # vitest 한 번 실행
npm run test:watch   # watch 모드
```

순수 모듈(core / render / util / ui / app/model-source) 대상 단위 테스트만 있다. Canvas·DOM 통합은 수동 브라우저 스모크로 확인한다.

## 모델 교체

- 기본 모델: `public/best.onnx` 덮어쓰기.
- 임시 교체: 앱 설정 시트에서 `.onnx` 파일을 직접 업로드.
- 루트 `.gitignore` 는 `*.onnx` 를 무시하지만 `public/best.onnx` 는 화이트리스트 처리되어 있다.

## 추론 백엔드

설정 시트에서 WebGPU / WASM 전환이 가능하다. 지원 여부는 기기에 따라 다르며, iOS Safari 등은 저메모리 모드로 자동 전환된다(`src/util/device.ts`).
