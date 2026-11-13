//! The sources the runner keeps current, each as the commands its first load used.

/// One build command whose run folder is diffed and published on its own.
pub struct Build {
    pub name: &'static str,
    pub argv: &'static [&'static str],
    /// The build lists every page the source has, so a page missing from it is gone.
    pub full_list: bool,
    /// Pages carry live numbers, so most change every run and a large share changing is no alarm.
    pub volatile: bool,
    /// Text lines starting with one of these are left out of a page's hash, such as a credit line naming the run's date.
    pub volatile_lines: &'static [&'static str],
    /// Columns left out of a page's hash, such as a modified date every run sets.
    pub volatile_columns: &'static [&'static str],
}

pub struct Source {
    pub name: &'static str,
    pub fetch: &'static [&'static [&'static str]],
    pub builds: &'static [Build],
    /// Folder the commands run in; the subnet-22 checkout when empty.
    pub cwd: &'static str,
    /// Interpreter `{python}` names; the runner's bulk venv when empty.
    pub python: &'static str,
    /// Address-space cap for each command in bytes; none when 0.
    pub memory_bytes: u64,
    pub timeout_s: u64,
}

pub const REPO: &str = "~/refresh/subnet-22";
pub const PYTHON: &str = "~/refresh/venv/bin/python";
/// The Rust bulk builders (subnet-22 `bulk-rs`), for the sources they cover.
const BULK: &str = "~/refresh/bin/bulk";
/// desearch-bot, whose `refresh edgar|fred` are the crawler's refresh commands in Rust.
const BOT: &str = "~/refresh/bin/desearch-bot";
const CRAWLER: &str = "~/desearch-engine.v8";
const CRAWLER_PYTHON: &str = "~/venv/bin/python";
const CRAWLER_MEMORY: u64 = 4 << 30;
const HOUR: u64 = 3600;
/// EUR-Lex texts kept between runs, so a run fetches only the acts and judgments it has not saved.
const EURLEX_CACHE: &str = "~/refresh/cache/eurlex";

pub const SOURCES: &[Source] = &[
    Source {
        name: "bittensor",
        fetch: &[&[BULK, "bittensor-fetch", "--out", "{work}/src", "--python", "{python}"]],
        builds: &[
            Build {
                name: "subnets",
                argv: &[BULK, "subnets", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/subnets-chain.json"],
                full_list: true,
                volatile: true,
                volatile_lines: &[],
                volatile_columns: &[],
            },
            Build {
                name: "subnet-sites",
                argv: &[BULK, "subnet-sites", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--workers", "4", "--python", "{python}", "{work}/src/sites"],
                full_list: true,
                volatile: false,
                volatile_lines: &[],
                volatile_columns: &[],
            },
            Build {
                name: "subnet-github",
                argv: &[BULK, "subnet-github", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--workers", "4", "--python", "{python}", "{work}/src/github"],
                full_list: true,
                volatile: false,
                volatile_lines: &[],
                volatile_columns: &[],
            },
        ],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "arxiv",
        fetch: &[&[BULK, "arxiv-harvest", "--out", "{work}/oai", "--from", "{since}", "--until", "{until}"]],
        builds: &[Build {
            name: "arxiv",
            argv: &[BULK, "arxiv", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/oai/*.xml"],
            full_list: false,
            volatile: false,
            volatile_lines: &[],
            volatile_columns: &[],
        }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "edgar",
        fetch: &[],
        builds: &[Build {
            name: "edgar",
            argv: &[BOT, "refresh", "edgar", "--since", "{since}", "--until", "{until}", "--out", "{out}", "--per-second", "8", "--python", "{python}"],
            full_list: false,
            volatile: false,
            volatile_lines: &[],
            volatile_columns: &[],
        }],
        cwd: CRAWLER,
        python: CRAWLER_PYTHON,
        memory_bytes: CRAWLER_MEMORY,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "fred",
        fetch: &[],
        builds: &[Build { name: "fred", argv: &[BOT, "refresh", "fred", "--out", "{out}", "--python", "{python}"], full_list: false, volatile: true, volatile_lines: &[], volatile_columns: &[] }],
        cwd: CRAWLER,
        python: CRAWLER_PYTHON,
        memory_bytes: CRAWLER_MEMORY,
        timeout_s: 80 * HOUR,
    },
    Source {
        name: "medlineplus",
        fetch: &[&[BULK, "medlineplus-fetch", "--out", "{work}/src"]],
        builds: &[Build {
            name: "medlineplus",
            argv: &[BULK, "medlineplus", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--python", "{python}", "--topics", "{work}/src/topics.xml", "--genetics", "{work}/src/genetics.xml"],
            full_list: true,
            volatile: false,
            volatile_lines: &[],
            volatile_columns: &["modified"],
        }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 2 * HOUR,
    },
    Source {
        name: "uscode",
        fetch: &[&[BULK, "uscode-fetch", "--out", "{work}/src"]],
        builds: &[Build { name: "uscode", argv: &[BULK, "uscode", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/xml_uscAll@*.zip"], full_list: true, volatile: false, volatile_lines: &["Source: United States Code"], volatile_columns: &[] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 2 * HOUR,
    },
    Source {
        name: "ecfr",
        fetch: &[&[BULK, "ecfr-fetch", "--out", "{work}/src"]],
        builds: &[Build { name: "ecfr", argv: &[BULK, "ecfr", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--titles", "{work}/src/titles.json", "{work}/src"], full_list: true, volatile: false, volatile_lines: &["Source: Electronic Code of Federal Regulations"], volatile_columns: &["modified"] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 4 * HOUR,
    },
    Source {
        name: "fedreg",
        fetch: &[&[BULK, "fedreg-fetch", "--out", "{work}/src", "--from", "{since}", "--to", "{until}"]],
        builds: &[Build { name: "fedreg", argv: &[BULK, "fedreg", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--index", "{work}/src/index", "{work}/src/FR-*.xml"], full_list: false, volatile: false, volatile_lines: &[], volatile_columns: &[] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 4 * HOUR,
    },
    Source {
        name: "eurlex",
        fetch: &[&[BULK, "eurlex-fetch", "--out", EURLEX_CACHE, "--relist", "--per-second", "3"]],
        builds: &[Build { name: "eurlex", argv: &[BULK, "eurlex", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--python", "{python}", EURLEX_CACHE], full_list: true, volatile: false, volatile_lines: &[], volatile_columns: &[] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 24 * HOUR,
    },
    Source {
        name: "dailymed",
        fetch: &[&[BULK, "dailymed-fetch", "--out", "{work}/src", "--since", "{since}"]],
        builds: &[Build { name: "dailymed", argv: &[BULK, "dailymed", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/*.zip"], full_list: false, volatile: false, volatile_lines: &[], volatile_columns: &[] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 4 * HOUR,
    },
    Source {
        name: "ctgov",
        fetch: &[&[BULK, "download", "--to", "{work}/src/ctg-studies.json.zip", "https://clinicaltrials.gov/api/v2/studies/download?format=json.zip"]],
        builds: &[
            Build { name: "ctgov-a", argv: &[BULK, "ctgov", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--nct-to", "NCT02600000", "{work}/src/ctg-studies.json.zip"], full_list: false, volatile: false, volatile_lines: &["Source: ClinicalTrials.gov"], volatile_columns: &[] },
            Build {
                name: "ctgov-b",
                argv: &[BULK, "ctgov", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--nct-from", "NCT02600000", "--nct-to", "NCT05300000", "{work}/src/ctg-studies.json.zip"],
                full_list: false,
                volatile: false,
                volatile_lines: &["Source: ClinicalTrials.gov"],
                volatile_columns: &[],
            },
            Build { name: "ctgov-c", argv: &[BULK, "ctgov", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--nct-from", "NCT05300000", "{work}/src/ctg-studies.json.zip"], full_list: false, volatile: false, volatile_lines: &["Source: ClinicalTrials.gov"], volatile_columns: &[] },
        ],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "uniprot",
        fetch: &[
            &[BULK, "download", "--to", "{work}/src/uniprot_sprot.xml.gz", "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz"],
            &[BULK, "download", "--to", "{work}/src/reldate.txt", "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/reldate.txt"],
        ],
        builds: &[
            Build { name: "uniprot-a", argv: &[BULK, "uniprot", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--to", "200000", "{work}/src/uniprot_sprot.xml.gz"], full_list: false, volatile: false, volatile_lines: &["Source: UniProtKB/Swiss-Prot"], volatile_columns: &[] },
            Build {
                name: "uniprot-b",
                argv: &[BULK, "uniprot", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--from", "200000", "--to", "400000", "{work}/src/uniprot_sprot.xml.gz"],
                full_list: false,
                volatile: false,
                volatile_lines: &["Source: UniProtKB/Swiss-Prot"],
                volatile_columns: &[],
            },
            Build { name: "uniprot-c", argv: &[BULK, "uniprot", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--from", "400000", "{work}/src/uniprot_sprot.xml.gz"], full_list: false, volatile: false, volatile_lines: &["Source: UniProtKB/Swiss-Prot"], volatile_columns: &[] },
        ],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "pubchem",
        fetch: &[&[BULK, "pubchem-fetch", "--out", "{work}/src", "--delay", "5"]],
        builds: &[Build { name: "pubchem", argv: &[BULK, "pubchem", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src"], full_list: true, volatile: false, volatile_lines: &[], volatile_columns: &[] }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "gutenberg",
        fetch: &[
            &[BULK, "download", "--to", "{work}/src/pg_catalog.csv", "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"],
            &[BULK, "download", "--to", "{work}/src/rdf-files.tar.bz2", "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"],
            &["mkdir", "-p", "{work}/src/rdf"],
            &["tar", "-xjf", "{work}/src/rdf-files.tar.bz2", "-C", "{work}/src/rdf"],
            &[BULK, "download", "--to", "{work}/src/txt-files.tar.zip", "https://www.gutenberg.org/cache/epub/feeds/txt-files.tar.zip"],
        ],
        builds: &[Build {
            name: "gutenberg",
            argv: &[BULK, "gutenberg", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "--catalog", "{work}/src/pg_catalog.csv", "--rdf", "{work}/src/rdf", "{work}/src/txt-files.tar.zip"],
            full_list: true,
            volatile: false,
            volatile_lines: &[],
            volatile_columns: &[],
        }],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
];

pub fn find(name: &str) -> Option<&'static Source> {
    SOURCES.iter().find(|source| source.name == name)
}
