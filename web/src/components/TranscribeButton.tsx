import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useLocation } from 'wouter'

import { Button } from './Button'
import { Select } from './Select'
import { useToast } from './Toast'
import { useApiMutation, useApiQuery, useSession } from '@/hooks'
import { apiPost } from '@/lib/api'
import { transcriptHref } from '@/lib/contentLinks'
import { paths } from '@/lib/endpoints'
import { atLeast } from '@/lib/roles'

/** A transcription a listing says already exists for the download. */
export interface ExistingTranscription {
  transcription_id: string
  state: string
  engine?: TranscriptionEngine
}

export type TranscriptionEngine = 'local' | 'qwen'

export interface TranscriberEngineHealth {
  ready: boolean
  model: string | null
  workers: number
  running: number
  queued: number
}

export interface TranscribeButtonProps {
  /** The download holding the video, or null when there is none yet. */
  downloadId: string | null
  /** Why it cannot be transcribed now, shown under the button; null when it can. */
  unavailable?: string | null
  /**
   * The transcript this download already has, when the listing said so. The
   * button opens it rather than offering to start one: the API reuses a
   * finished transcript unless asked to refresh, so a click could only ever
   * be handed back this record.
   */
  existing?: ExistingTranscription | null
}

/**
 * Whether this instance can transcribe at all, from `components.transcriber`.
 *
 * `off` is the profile not being enabled, which is the default - the button
 * was always enabled and every click answered TRANSCRIBER_UNAVAILABLE. `down`
 * is configured but unreachable or has no ready engine. `unknown` covers a
 * server that does not report the component (older builds) and a status read
 * that failed; nothing is blocked on it, so the button behaves as it always did.
 */
export type TranscriberState = 'ready' | 'off' | 'down' | 'unknown'

interface StatusComponents {
  components?: Record<
    string,
    | {
        configured?: boolean
        ok?: boolean | null
        engines?: Partial<Record<TranscriptionEngine, TranscriberEngineHealth>>
      }
    | undefined
  >
}

export type TranscriberEngineState =
  | 'available'
  | 'unavailable'
  | 'unknown'
  | 'not_configured'
  | 'unreachable'

export interface TranscriberStatus {
  state: TranscriberState
  engines: Partial<Record<TranscriptionEngine, TranscriberEngineHealth>>
  isLoading: boolean
  isError: boolean
  hasData: boolean
  refresh: () => void
}

export function useTranscriberStatus(): TranscriberStatus {
  // Read when it is needed rather than polled: the status call probes every
  // component, and whether a compose profile is on changes by redeploying,
  // not while a drawer is open.
  const status = useApiQuery<StatusComponents>({
    key: ['system', 'status'],
    path: paths.system.status,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const component = status.data?.components?.['transcriber']
  let state: TranscriberState = 'unknown'
  if (component && typeof component.configured === 'boolean') {
    if (!component.configured) state = 'off'
    else if (component.ok === false) state = 'down'
    else if (component.ok === true) state = 'ready'
  }
  return {
    state,
    engines: component?.engines ?? {},
    isLoading: status.isLoading,
    isError: status.isError,
    hasData: status.data !== undefined,
    refresh: () => {
      void status.refetch()
    },
  }
}

export function useTranscriberState(): TranscriberState {
  return useTranscriberStatus().state
}

export function transcriberEngineState(
  status: Pick<TranscriberStatus, 'state' | 'engines'>,
  engine: TranscriptionEngine,
): TranscriberEngineState {
  const health = status.engines[engine]
  if (health) return health.ready ? 'available' : 'unavailable'
  if (status.state === 'off') return 'not_configured'
  if (status.state === 'down') return 'unreachable'
  return 'unknown'
}

/** The installation guide's section on the optional profiles, per language. */
const PROFILES_GUIDE = {
  en: 'https://github.com/Evil0ctal/Douyin_TikTok_Download_API/blob/main/documents/en/02-installation.md#the-optional-profiles',
  // The heading is Chinese, so its anchor is too: percent-encoded here
  // because source files stay ASCII.
  zh: 'https://github.com/Evil0ctal/Douyin_TikTok_Download_API/blob/main/documents/zh/02-installation.md#%E5%8F%AF%E9%80%89-profile',
} as const

export function transcriberGuideHref(language: string): string {
  return language.startsWith('zh') ? PROFILES_GUIDE.zh : PROFILES_GUIDE.en
}

/**
 * Why nothing can be transcribed, and what turns it on. Null while it can be.
 *
 * Shared by the button and by the Transcriptions page, so the two cannot give
 * different instructions for the same switch.
 */
export function TranscriberNotice({ state }: { state: TranscriberState }) {
  const { t, i18n } = useTranslation(['console', 'common'])
  if (state !== 'off' && state !== 'down') return null
  return (
    <p className="u-xs u-muted">
      {t(`transcriptions.transcriber.${state}Body`)}{' '}
      {state === 'off' ? (
        <a href={transcriberGuideHref(i18n.language)} target="_blank" rel="noreferrer noopener">
          {t('transcriptions.transcriber.guide')}
        </a>
      ) : null}
    </p>
  )
}

/**
 * Start, or reuse, the transcription of a stored video, then open it.
 *
 * One component for the Downloads drawer and the Library drawer, so the two
 * send the same request and land on the same page. The API reuses a finished
 * transcription of the same file unless asked to refresh, so a second click
 * opens the first rather than running the model again.
 *
 * Hidden below viewer. The demo account may open both pages, but
 * `POST /transcriptions` is not on its write allowlist, so the button could
 * only ever fail for it - and a success would land it on /transcriptions,
 * which it cannot open either.
 */
export function TranscribeButton({
  downloadId,
  unavailable = null,
  existing = null,
}: TranscribeButtonProps) {
  const { t } = useTranslation(['console', 'common'])
  const toast = useToast()
  const [, navigate] = useLocation()
  const session = useSession()
  const transcriberStatus = useTranscriberStatus()
  const transcriber = transcriberStatus.state
  const [engine, setEngine] = useState<TranscriptionEngine>('local')

  const transcribe = useApiMutation<
    { transcription_id: string },
    { id: string; engine: TranscriptionEngine }
  >(
    ({ id, engine: selectedEngine }) =>
      apiPost(
        paths.transcriptions.create,
        { download_id: id, engine: selectedEngine, normalize: true, refresh: false },
        { awaitTask: false },
      ),
    {
      onSuccess: (result) => {
        navigate(transcriptHref(result.transcription_id))
      },
      onError: (error) => toast.apiError(error, t('downloads.transcription.failed')),
    },
  )

  if (!atLeast(session.data?.role ?? null, 'viewer')) return null

  const blocked = transcriber === 'off' || transcriber === 'down'
  const selectedEngineState = transcriberEngineState(transcriberStatus, engine)
  const selectedEngineUnavailable = ['unavailable', 'not_configured', 'unreachable'].includes(
    selectedEngineState,
  )
  const engineReason =
    selectedEngineState === 'unavailable'
      ? t('downloads.transcription.engineUnavailable')
      : null
  const reason =
    unavailable ??
    (blocked ? t(`transcriptions.transcriber.${transcriber}Title`) : engineReason)
  // Older servers did not attach the engine to download listings; those
  // records predate Qwen and are therefore local.
  const existingEngine = existing?.engine ?? 'local'
  const opensExisting = existing !== null && existingEngine === engine

  return (
    <div className="u-stack-sm">
      <Select
        label={t('downloads.transcription.engine')}
        description={t(`downloads.transcription.${engine}Hint`)}
        value={engine}
        disabled={transcribe.isPending}
        onChange={(event) => setEngine(event.target.value as TranscriptionEngine)}
        options={[
          {
            value: 'local',
            label: t('downloads.transcription.local'),
            disabled: ['unavailable', 'not_configured', 'unreachable'].includes(
              transcriberEngineState(transcriberStatus, 'local'),
            ),
          },
          {
            value: 'qwen',
            label: t('downloads.transcription.qwen'),
            disabled: ['unavailable', 'not_configured', 'unreachable'].includes(
              transcriberEngineState(transcriberStatus, 'qwen'),
            ),
          },
        ]}
      />
      <Button
        variant="primary"
        loading={transcribe.isPending}
        disabled={
          !opensExisting &&
          (downloadId === null || reason !== null || selectedEngineUnavailable || transcribe.isPending)
        }
        title={!opensExisting ? (reason ?? undefined) : undefined}
        onClick={() => {
          if (opensExisting && existing) {
            navigate(transcriptHref(existing.transcription_id))
          } else if (downloadId !== null) {
            transcribe.mutate({ id: downloadId, engine })
          }
        }}
      >
        {opensExisting && existing
          ? existing.state === 'done'
            ? t('downloads.transcription.open')
            : t('downloads.transcription.openRunning')
          : t('downloads.transcription.action')}
      </Button>
      {unavailable ? (
        <p className="u-xs u-muted">{unavailable}</p>
      ) : engineReason && !opensExisting ? (
        <p className="u-xs u-muted">{engineReason}</p>
      ) : (
        <TranscriberNotice state={transcriber} />
      )}
    </div>
  )
}
