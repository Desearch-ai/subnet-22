import { Table, Td, Th, Tr } from '@/components/ui/Table'
import { formatClock, formatInt } from '@/lib/format'
import { HOUR_S, type HourSeries } from './hourSeries'

interface HourTableProps<P extends { at: number }> {
  points: readonly P[]
  /** The last point is the hour still running. */
  running?: boolean
  columns: readonly { series: readonly HourSeries<P>[]; format?: (value: number) => string }[]
}

/** The charts' hours as rows, newest first. */
export function HourTable<P extends { at: number }>({
  points,
  running = false,
  columns,
}: HourTableProps<P>) {
  const cells = columns.flatMap(({ series, format = formatInt }) =>
    series.map((item) => ({ item, format })),
  )
  return (
    <Table>
      <thead>
        <tr>
          <Th>Hour</Th>
          {cells.map(({ item }) => (
            <Th key={item.key} align="right">
              {item.label}
            </Th>
          ))}
        </tr>
      </thead>
      <tbody>
        {[...points].reverse().map((point, index) => (
          <Tr key={point.at}>
            <Td className="font-mono text-xs">
              {formatClock(point.at)}–
              {running && index === 0 ? 'now' : formatClock(point.at + HOUR_S)}
            </Td>
            {cells.map(({ item, format }) => (
              <Td key={item.key} numeric>
                {format(item.value(point))}
              </Td>
            ))}
          </Tr>
        ))}
      </tbody>
    </Table>
  )
}
