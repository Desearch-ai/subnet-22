-- The shared list of domains: what each is, which bucket owns it, and what its last visit found.

CREATE SCHEMA IF NOT EXISTS bot;
SET search_path TO bot;

-- A domain's crawl state lives in the store of the process that owns its bucket.
CREATE TABLE IF NOT EXISTS domains (
    host           text PRIMARY KEY,
    bucket         smallint NOT NULL,
    rank           integer,
    tld_group      text,

    state          text NOT NULL DEFAULT 'new',
    state_reason   text,
    checked_at     timestamptz,
    url_count      bigint NOT NULL DEFAULT 0,
    canonical_host text,

    category       text,
    categories     text[],
    changed_at     timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT domains_state_check CHECK (state IN (
        'new', 'active', 'failing', 'down', 'unreachable',
        'no_sitemap', 'redirects', 'blocked', 'ineligible', 'excluded'))
);

-- Each crawler asks only for what changed in its own buckets since it last looked.
CREATE INDEX IF NOT EXISTS domains_changes_idx ON domains (bucket, changed_at);
CREATE INDEX IF NOT EXISTS domains_categories_idx ON domains USING gin (categories);
CREATE INDEX IF NOT EXISTS domains_canonical_idx ON domains (canonical_host)
    WHERE canonical_host IS NOT NULL;

-- One row per source per label, so a refresh from one source never discards another's verdict.
CREATE TABLE IF NOT EXISTS domain_categories (
    host           text NOT NULL REFERENCES domains(host) ON DELETE CASCADE,
    source         text NOT NULL,
    category       text NOT NULL,
    raw_category   text,
    category_id    integer,
    super_category text,
    recorded_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (host, source, category)
);

CREATE INDEX IF NOT EXISTS domain_categories_category_idx ON domain_categories (category);

-- When each source last looked at a domain, whether or not it had anything to say.
CREATE TABLE IF NOT EXISTS category_checks (
    host       text NOT NULL REFERENCES domains(host) ON DELETE CASCADE,
    source     text NOT NULL,
    checked_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (host, source)
);

CREATE TABLE IF NOT EXISTS excluded_categories (
    category text PRIMARY KEY,
    reason   text NOT NULL
);

INSERT INTO excluded_categories (category, reason) VALUES
    ('adult',         'sexual content'),
    ('gambling',      'casinos and betting'),
    ('bank',          'login screens with nothing to index, and probing them reads as a scan'),
    ('malware',       'unsafe'),
    ('phishing',      'unsafe'),
    ('cryptojacking', 'unsafe'),
    ('stalkerware',   'unsafe'),
    ('ddos',          'unsafe'),
    ('hacking',       'unsafe'),
    ('warez',         'unlicensed distribution'),
    ('dangerous',     'instructions for causing harm'),
    ('dialer',        'unsafe'),
    ('cheating',      'academic fraud'),
    ('shortener',     'no content of its own'),
    ('redirector',    'no content of its own'),
    ('ads',           'advertising and tracking endpoints'),
    ('marketingware', 'advertising and tracking endpoints'),
    ('dynamic_dns',   'infrastructure, not a publisher'),
    ('doh',           'infrastructure, not a publisher'),
    ('proxy',         'infrastructure, not a publisher'),
    ('social',        'pages generated per user rather than published'),
    ('forums',        'pages generated per user rather than published'),
    ('chat',          'pages generated per user rather than published'),
    ('webmail',       'pages generated per user rather than published'),
    ('filehosting',   'pages generated per user rather than published')
ON CONFLICT (category) DO NOTHING;

-- Exactly what is published: not known to be bad, and carrying no disqualifying category.
CREATE OR REPLACE VIEW published_domains AS
    SELECT host FROM domains
    WHERE state IN ('new', 'active', 'failing', 'no_sitemap')
      AND NOT (coalesce(categories, '{}') && ARRAY(SELECT category FROM excluded_categories))
    ORDER BY host;
