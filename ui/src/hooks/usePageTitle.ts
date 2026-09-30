import { useEffect } from 'react'

const SITE_NAME = 'Desearch Dashboard'

export function usePageTitle(title: string) {
  useEffect(() => {
    document.title = `${SITE_NAME} - ${title}`
  }, [title])
}
