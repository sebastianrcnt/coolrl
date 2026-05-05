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

## Advisory workflow

Lost Cities Deep CFR처럼 실험 해석, 학습 안정성, 알고리즘 설계 판단이 애매한 작업에서는 필요할 때 Claude Opus 4.7 xhigh thinking에게 MCP로 2차 의견을 구할 수 있다.

Guidelines:
- Claude도 이 repository에 직접 접근할 수 있다고 가정하고, 관련 config, 코드 경로, checkpoint/metrics 경로, 현재 해석, 판단이 필요한 질문을 간결하게 전달한다.
- 자문은 의사결정 보조용이다. 최종 변경은 로컬에서 코드, 테스트, metrics를 직접 확인한 뒤 적용한다.
- 단순 구현 오류, 명확한 테스트 실패, 작은 리팩터링처럼 로컬에서 바로 판정 가능한 문제에는 자문을 기본값으로 쓰지 않는다.
- 자문 결과를 따를 때는 어떤 근거로 채택했는지와 어떤 부분은 보류했는지를 작업 요약에 남긴다.
