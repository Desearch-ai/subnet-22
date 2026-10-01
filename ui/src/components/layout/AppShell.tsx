import { NavLink, Outlet } from 'react-router'
import desearchLogo from '@/assets/desearch-logo.png'
import { cn } from '@/lib/cn'
import { MINERS_PATH, OVERVIEW_PATH, TASKS_PATH, VALIDATORS_PATH } from '@/lib/paths'
import { BackoffNotice } from './BackoffNotice'

const NAV_ITEMS = [
  { to: OVERVIEW_PATH, label: 'Overview' },
  { to: TASKS_PATH, label: 'Tasks' },
  { to: MINERS_PATH, label: 'Miners' },
  { to: VALIDATORS_PATH, label: 'Validators' },
] as const

export function AppShell() {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="border-line bg-ground sticky top-0 z-30 border-b">
        <div className="mx-auto flex max-w-[88rem] items-center gap-6 px-4 py-2.5">
          <NavLink
            to={OVERVIEW_PATH}
            aria-label="Desearch Dashboard"
            className="flex shrink-0 items-center"
          >
            <img src={desearchLogo} alt="Desearch" width={320} height={62} className="h-5 w-auto" />
          </NavLink>
          <nav aria-label="Sections" className="scroll-thin flex min-w-0 gap-1 overflow-x-auto">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === OVERVIEW_PATH}
                className={({ isActive }) =>
                  cn(
                    'rounded-md px-2.5 py-1 text-sm whitespace-nowrap transition-colors',
                    isActive ? 'bg-raised text-ink' : 'text-ink-muted hover:text-ink',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
        <BackoffNotice />
      </header>
      <main className="mx-auto w-full max-w-[88rem] flex-1 px-4 py-5">
        <Outlet />
      </main>
    </div>
  )
}
