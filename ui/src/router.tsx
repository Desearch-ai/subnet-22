import { createBrowserRouter } from 'react-router'
import { AppShell } from '@/components/layout/AppShell'
import { RouteError } from '@/components/layout/RouteError'
import { RouteLoading } from '@/components/layout/RouteLoading'
import { MINERS_PATH, OVERVIEW_PATH, TASKS_PATH, VALIDATORS_PATH } from '@/lib/paths'

export const router = createBrowserRouter([
  {
    path: OVERVIEW_PATH,
    Component: AppShell,
    ErrorBoundary: RouteError,
    HydrateFallback: RouteLoading,
    children: [
      {
        index: true,
        lazy: async () => ({ Component: (await import('@/pages/OverviewPage')).OverviewPage }),
      },
      {
        path: TASKS_PATH,
        lazy: async () => ({ Component: (await import('@/pages/TasksPage')).TasksPage }),
      },
      {
        path: `${TASKS_PATH}/:taskId`,
        lazy: async () => ({ Component: (await import('@/pages/TaskDetailPage')).TaskDetailPage }),
      },
      {
        path: MINERS_PATH,
        lazy: async () => ({ Component: (await import('@/pages/MinersPage')).MinersPage }),
      },
      {
        path: `${MINERS_PATH}/:hotkey`,
        lazy: async () => ({
          Component: (await import('@/pages/MinerDetailPage')).MinerDetailPage,
        }),
      },
      {
        path: VALIDATORS_PATH,
        lazy: async () => ({ Component: (await import('@/pages/ValidatorsPage')).ValidatorsPage }),
      },
      {
        path: `${VALIDATORS_PATH}/:hotkey`,
        lazy: async () => ({
          Component: (await import('@/pages/ValidatorDetailPage')).ValidatorDetailPage,
        }),
      },
      {
        path: '*',
        lazy: async () => ({ Component: (await import('@/pages/NotFoundPage')).NotFoundPage }),
      },
    ],
  },
])
