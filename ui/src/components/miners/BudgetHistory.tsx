import type { BudgetTransition } from '@/api/types'
import { Identifier } from '@/components/ui/Identifier'
import { EmptyState } from '@/components/ui/States'
import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { TimeAgo } from '@/components/ui/TimeAgo'
import { EMPTY_VALUE } from '@/lib/format'
import { BUDGET_CAUSE_MEANING } from '@/lib/labels'
import { taskPath } from '@/lib/paths'

export function BudgetHistory({ transitions }: { transitions: readonly BudgetTransition[] }) {
  if (transitions.length === 0) {
    return (
      <EmptyState title="The budget has not changed yet.">
        It grows when uploads pass and shrinks when they fail or a claim runs out.
      </EmptyState>
    )
  }
  return (
    <div className="scroll-thin max-h-96 overflow-y-auto">
      <Table>
        <thead>
          <tr>
            <Th>When</Th>
            <Th>Kind</Th>
            <Th align="right">Budget</Th>
            <Th>Why</Th>
            <Th>Task</Th>
          </tr>
        </thead>
        <tbody>
          {transitions.map((step) => (
            <Tr key={`${step.at}:${step.pool}:${step.old}:${step.new}`}>
              <Td className="text-ink-muted text-xs">
                <TimeAgo at={step.at} />
              </Td>
              <Td className="font-mono text-xs">{step.pool}</Td>
              <Td numeric>
                <span className="text-ink-faint">{step.old} → </span>
                {step.new}
              </Td>
              <Td className="text-ink-muted">{BUDGET_CAUSE_MEANING[step.cause]}</Td>
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
    </div>
  )
}
