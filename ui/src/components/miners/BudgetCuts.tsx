import type { BudgetTransition, Kind } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { EMPTY_VALUE } from '@/lib/format'
import { BUDGET_CAUSE_MEANING } from '@/lib/labels'
import { taskPath } from '@/lib/paths'

/** Only the steps that lowered the budget: raises come one task at a time and say little. */
export function BudgetCuts({
  transitions,
  kind,
}: {
  transitions: readonly BudgetTransition[]
  kind: Kind
}) {
  const cuts = transitions.filter((step) => step.pool === kind && step.new < step.old)
  if (cuts.length === 0) {
    return <EmptyState title={`No recent cut to the ${kind} budget.`} />
  }
  return (
    <Table>
      <thead>
        <tr>
          <Th>When</Th>
          <Th>Why</Th>
          <Th align="right">Budget</Th>
          <Th>Task</Th>
        </tr>
      </thead>
      <tbody>
        {cuts.map((step) => (
          <Tr key={`${step.at}:${step.old}:${step.new}`}>
            <Td className="text-ink-muted text-xs">
              <TimeAgo at={step.at} />
            </Td>
            <Td className="text-ink-muted">{BUDGET_CAUSE_MEANING[step.cause]}</Td>
            <Td numeric>
              <span className="text-ink-faint">{step.old} → </span>
              {step.new}
            </Td>
            <Td>
              {step.task_id ? (
                <Identifier value={step.task_id} label="task id" to={taskPath(step.task_id)} />
              ) : (
                <span className="text-ink-faint">{EMPTY_VALUE}</span>
              )}
            </Td>
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
