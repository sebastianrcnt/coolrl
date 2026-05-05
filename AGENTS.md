## Repository language policy

This repository uses Korean as the default language for project communication.

Required:
- Commit messages must be written in Korean.
- PR titles and PR descriptions must be written in Korean.
- GitHub issue comments and review comments must be written in Korean.
- Documentation changes must be written in Korean unless the document is intentionally maintained in English.
- Changelog and release notes must be written in Korean.

Exceptions:
- Code identifiers, filenames, package names, command names, CLI flags, config keys, protocol fields, and public API names should remain in English.
- Do not translate quoted external errors, log messages, dependency names, or upstream API names.
- Preserve the language of existing English documentation unless the task explicitly asks to translate it.

## Commit message policy

Commit messages must be written in Korean. Use this structure by default:

```text
<한 줄 요약>

맥락:
- 왜 바꾸는지, 어떤 실험 해석이나 운영 판단이 있는지

변경:
- 실제 수정 내용

확인:
- 실행한 테스트, 확인한 metrics, 링크, checkpoint 상태
```

Keep small commits compact, but do not leave the body empty for commits that affect experiment settings, results, algorithm decisions, evaluation criteria, checkpoint/data retention policy, or documentation archive decisions.

For those research/operation commits, write enough concrete detail that `git show` alone can recover the decision:
- `맥락`: 관찰한 문제, 변경을 유도한 실험 결과, 실패 모드, 운영 제약을 적는다.
- `변경`: 바꾼 config/path/code/doc 동작과 중요한 이름, 값, 경로를 적는다.
- `확인`: 실행한 명령, 확인한 metric, checkpoint/link 상태, 또는 검증하지 못한 이유를 적는다.

Avoid generic filler such as "문서 수정" or "테스트 확인" when a concrete path, metric, or command is available.

## Advisory workflow

Lost Cities Deep CFR처럼 실험 해석, 학습 안정성, 알고리즘 설계 판단이 애매한 작업에서는 필요할 때 MCP를 통해 Claude Opus 4.7 xhigh thinking에게 2차 의견을 구할 수 있다.

Guidelines:
- Claude도 이 repository에 직접 접근 권한이 있다고 가정한다. 긴 코드/문서 내용을 복사해 붙이기보다 관련 config, 코드 경로, checkpoint/metrics 경로, 현재 해석, 판단이 필요한 질문을 간결하게 전달한다.
- 자문 요청에는 가능한 한 실행 중인 실험 이름, 주요 metrics, 평가 command/path, 비교해야 할 후보를 포함한다.
- 자문은 의사결정 보조용이다. 최종 변경은 로컬에서 코드, 테스트, metrics를 직접 확인한 뒤 적용한다.
- 단순 구현 오류, 명확한 테스트 실패, 작은 리팩터링처럼 로컬에서 바로 판정 가능한 문제에는 자문을 기본값으로 쓰지 않는다.
- 자문 결과를 따를 때는 어떤 근거로 채택했는지와 어떤 부분은 보류했는지를 작업 요약에 남긴다.
