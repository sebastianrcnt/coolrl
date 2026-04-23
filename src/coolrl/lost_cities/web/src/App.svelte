<script lang="ts">
  type ExpeditionColor = "yellow" | "blue" | "white" | "green" | "red";

  const colors: ExpeditionColor[] = ["yellow", "blue", "white", "green", "red"];
  const labels: Record<ExpeditionColor, string> = {
    yellow: "사막",
    blue: "바다",
    white: "빙하",
    green: "정글",
    red: "화산",
  };
  const accents: Record<ExpeditionColor, string> = {
    yellow: "#d7a03d",
    blue: "#4d7ff5",
    white: "#d9dde8",
    green: "#2fa56f",
    red: "#df6257",
  };

  const formatScore = (score: number): string => (score > 0 ? `+${score}` : `${score}`);

  let players = $state(2);
  let investments = $state(2);
  let handSize = $state(8);
  let selected = $state<ExpeditionColor>("yellow");

  const preview = $derived.by(() =>
    colors.map((color, index) => {
      const cards = Math.max(2, handSize - index);
      const multiplier = color === selected ? 1 + investments : 1;
      const score = cards * 4 * multiplier - 20;

      return {
        color,
        label: labels[color],
        cards,
        multiplier,
        score,
        active: color === selected,
      };
    }),
  );

  const activePlan = $derived(preview.find((plan) => plan.active) ?? preview[0]!);
  const bestScore = $derived(Math.max(...preview.map((plan) => plan.score)));
  const summary = $derived(
    activePlan.score >= 0 ? "바로 출발해도 되는 조합" : "조금 더 준비가 필요한 조합",
  );

  $effect(() => {
    document.title = `Lost Cities | ${labels[selected]} 원정`;
  });
</script>

<svelte:head>
  <meta
    name="description"
    content="Lost Cities용 Svelte 5 runes + TypeScript 스타터 프로젝트"
  />
</svelte:head>

<main class="shell">
  <section class="hero panel">
    <p class="eyebrow">Svelte 5 runes starter</p>
    <h1>Lost Cities web starter</h1>
    <p class="lede">
      `lost_cities/web` 아래에 바로 붙여 쓸 수 있는 최소 프로젝트입니다. 상태는
      <code>$state</code>, 계산값은 <code>$derived</code>, 부수 효과는 <code>$effect</code>로
      연결돼 있습니다.
    </p>

    <div class="headline-stats">
      <div>
        <span>플레이어</span>
        <strong>{players}명</strong>
      </div>
      <div>
        <span>투자 카드</span>
        <strong>{investments}장</strong>
      </div>
      <div>
        <span>최고 예상 점수</span>
        <strong>{formatScore(bestScore)}</strong>
      </div>
    </div>
  </section>

  <section class="panel controls">
    <div class="section-head">
      <h2>시뮬레이션 프리셋</h2>
      <span>{labels[selected]} 원정 선택 중</span>
    </div>

    <label class="slider-row" for="players">
      <span>플레이어 수</span>
      <input id="players" type="range" min="2" max="4" bind:value={players} />
      <strong>{players}</strong>
    </label>

    <label class="slider-row" for="investments">
      <span>투자 카드</span>
      <input id="investments" type="range" min="1" max="3" bind:value={investments} />
      <strong>{investments}</strong>
    </label>

    <label class="slider-row" for="hand-size">
      <span>손패 예상치</span>
      <input id="hand-size" type="range" min="5" max="8" bind:value={handSize} />
      <strong>{handSize}</strong>
    </label>
  </section>

  <section class="panel">
    <div class="section-head">
      <h2>원정 선택</h2>
      <span>색상 클릭으로 상태 변경</span>
    </div>

    <div class="chips">
      {#each colors as color}
        <button
          type="button"
          class:active={selected === color}
          style={`--chip:${accents[color]}`}
          onclick={() => {
            selected = color;
          }}
        >
          {labels[color]}
        </button>
      {/each}
    </div>
  </section>

  <section class="cards">
    {#each preview as plan}
      <article
        class:active={plan.active}
        class="panel plan-card"
        style={`--accent:${accents[plan.color]}`}
      >
        <div class="plan-header">
          <div class="plan-title">
            <span class="swatch"></span>
            <div>
              <p>{plan.label}</p>
              <small>{plan.cards}장 플레이 가정</small>
            </div>
          </div>
          <strong>{formatScore(plan.score)}</strong>
        </div>

        <div class="plan-meta">
          <span>x{plan.multiplier} 배수</span>
          {#if plan.active}
            <span class="badge">선택됨</span>
          {/if}
        </div>
      </article>
    {/each}
  </section>

  <section class="panel summary">
    <div>
      <p class="eyebrow">현재 프리뷰</p>
      <h2>{activePlan.label} 원정</h2>
      <p>{summary}</p>
    </div>

    <button
      type="button"
      class="primary"
      onclick={() => {
        handSize = Math.min(8, handSize + 1);
      }}
    >
      손패 한 장 추가
    </button>
  </section>
</main>
