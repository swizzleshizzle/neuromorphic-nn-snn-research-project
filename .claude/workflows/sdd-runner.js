export const meta = {
  name: 'sdd-runner',
  description: 'Execute an implementation plan task-by-task with implementer + reviewer + fix subagents, strictly sequential on the current branch.',
  whenToUse: 'When you have a written plan with self-contained tasks and want the subagent-driven loop automated. Pass args.base (branch base short hash), args.tasks ([{n, brief, report}]), and args.constraints (global constraints string).',
  phases: [{ title: 'Execute', detail: 'per task: implement -> review -> fix if blocking -> re-review' }],
}

const IMPL = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['DONE', 'BLOCKED'] },
    commit: { type: 'string', description: 'short HEAD hash after committing, or empty if BLOCKED' },
    testSummary: { type: 'string' },
    notes: { type: 'string' },
  },
  required: ['status', 'commit', 'testSummary'],
  additionalProperties: false,
}

const REVIEW = {
  type: 'object',
  properties: {
    specPass: { type: 'boolean' },
    qualityApproved: { type: 'boolean' },
    blocking: {
      type: 'array',
      items: {
        type: 'object',
        properties: { severity: { type: 'string' }, finding: { type: 'string' } },
        required: ['severity', 'finding'],
        additionalProperties: false,
      },
    },
    minor: { type: 'array', items: { type: 'string' } },
  },
  required: ['specPass', 'qualityApproved', 'blocking', 'minor'],
  additionalProperties: false,
}

const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const C = A.constraints || ''

function implPrompt(t, prev) {
  return `You are implementing ONE task (Task ${t.n}) of an implementation plan, strictly test-first.

Read your COMPLETE requirements (with the exact code to write) at: ${t.brief}

${C}

The previous task ended at commit ${prev}; build on top of it. Follow the brief's steps in order: write the failing test, run it to confirm it FAILS (RED), implement the minimal code, run the test to confirm it PASSES (GREEN), then commit (one commit for the task unless the brief explicitly says otherwise). Run tests with the repo venv, e.g. \`.venv/Scripts/python.exe -m pytest <paths> -v\`.

Write your full report (files changed, each command + its output, notes) to ${t.report}.

Return ONLY: status (DONE or BLOCKED); commit = the SHORT hash of HEAD after your commit (run \`git rev-parse --short HEAD\`); a one-line testSummary; and notes (any concerns, or "" if none). If you cannot get tests green, return BLOCKED with commit "" and explain in notes.`
}

function reviewPrompt(t, base, head, reReview) {
  return `You are ${reReview ? 'RE-reviewing (after a fix)' : 'reviewing'} Task ${t.n} of a plan. Give two verdicts: spec compliance and code quality.

Requirements (the task brief): ${t.brief}
Review the diff by running: \`git diff ${base} HEAD --stat\` then \`git diff ${base} HEAD\`. (HEAD is the branch tip after this task's commit.)

${C}

Verify the diff implements the brief (right signatures, real non-vacuous TDD tests for pure helpers, nothing extra) and that the binding constraints hold (the brain stays frozen; fixed-goal GridWorldEnv behavior is unchanged when new options are not passed; determinism by seed).

Return: specPass (bool), qualityApproved (bool), blocking = array of {severity, finding} for MUST-FIX-before-merge issues ONLY (Critical or Important; a real spec violation or a broken/vacuous test), and minor = array of strings for defer-able notes. Do not invent work. If it is clean, return blocking: [].`
}

function fixPrompt(t, base, blocking) {
  const list = blocking.map((b, i) => `${i + 1}. [${b.severity}] ${b.finding}`).join('\n')
  return `Address these blocking review findings on Task ${t.n} (work on the current git branch, do not switch branches):

${list}

${C}

Read the brief if needed: ${t.brief}. Make the MINIMAL fix, re-run the covering tests with \`.venv/Scripts/python.exe -m pytest <paths> -v\`, and commit. Append a fix note to ${t.report}.

Return ONLY: status (DONE or BLOCKED); commit = SHORT HEAD hash after your fix commit; a one-line testSummary; notes.`
}

phase('Execute')
const tasks = A.tasks || []
let prev = A.base
const ledger = []

if (!tasks.length) {
  return {
    error: 'no tasks parsed from args',
    argsType: typeof args,
    parsedKeys: (A && typeof A === 'object') ? Object.keys(A) : null,
    rawSample: (typeof args === 'string') ? args.slice(0, 120) : null,
  }
}

for (const t of tasks) {
  log(`Task ${t.n}: implementing (base ${prev})`)
  const impl = await agent(implPrompt(t, prev), { label: `t${t.n}-impl`, schema: IMPL, model: 'sonnet', phase: 'Execute' })
  if (!impl || impl.status !== 'DONE' || !impl.commit) {
    ledger.push({ task: t.n, status: impl ? impl.status : 'DIED', impl })
    log(`Task ${t.n}: not DONE (${impl ? impl.status : 'agent died'}) — stopping the run`)
    break
  }
  let head = impl.commit
  log(`Task ${t.n}: implemented at ${head}; reviewing`)
  let review = await agent(reviewPrompt(t, prev, head, false), { label: `t${t.n}-review`, schema: REVIEW, model: 'sonnet', phase: 'Execute' })
  let blocking = (review && review.blocking) || []

  if (blocking.length) {
    log(`Task ${t.n}: ${blocking.length} blocking finding(s) — dispatching fix`)
    const fix = await agent(fixPrompt(t, prev, blocking), { label: `t${t.n}-fix`, schema: IMPL, model: 'sonnet', phase: 'Execute' })
    if (fix && fix.commit) head = fix.commit
    review = await agent(reviewPrompt(t, prev, head, true), { label: `t${t.n}-rereview`, schema: REVIEW, model: 'sonnet', phase: 'Execute' })
    blocking = (review && review.blocking) || []
  }

  ledger.push({
    task: t.n,
    base: prev,
    head,
    specPass: review ? review.specPass : null,
    qualityApproved: review ? review.qualityApproved : null,
    blockingAfter: blocking,
    minor: review ? review.minor : [],
    testSummary: impl.testSummary,
  })
  log(`Task ${t.n}: done at ${head} (specPass=${review ? review.specPass : '?'}, blocking left=${blocking.length})`)
  prev = head
}

return ledger
