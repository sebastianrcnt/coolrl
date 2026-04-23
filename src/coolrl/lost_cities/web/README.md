# Lost Cities Web

`src/coolrl/lost_cities/web` 아래에 둔 Svelte 5 + TypeScript 스타터 프로젝트다.
기본 상태 관리는 runes(`$state`, `$derived`, `$effect`)로 작성되어 있어서
이후 게임 UI를 붙일 때 출발점으로 바로 쓸 수 있다.

## 요구 사항

- Node.js 20 이상
- npm 10 이상

## 설치

```bash
cd src/coolrl/lost_cities/web
npm install
```

## 개발 서버

```bash
npm run dev
```

기본적으로 `0.0.0.0` 바인딩이라 다른 기기에서도 접근 가능하다.

## 빌드

```bash
npm run build
npm run preview
```

`vite.config.ts` 는 개발 시 `/`, 프로덕션 빌드 시 `/coolrl/` base 를 사용한다.

## 타입 체크

```bash
npm run typecheck
```
