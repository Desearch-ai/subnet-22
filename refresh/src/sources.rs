//! The sources the runner keeps current, each as the commands its first load used.

/// One build command whose run folder is diffed and published on its own.
pub struct Build {
    pub name: &'static str,
    pub argv: &'static [&'static str],
    /// The build lists every page the source has, so a page missing from it is gone.
    pub full_list: bool,
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
const CRAWLER: &str = "~/desearch-engine.v8";
const CRAWLER_PYTHON: &str = "~/venv/bin/python";
const CRAWLER_MEMORY: u64 = 4 << 30;
const HOUR: u64 = 3600;

pub const SOURCES: &[Source] = &[
    Source {
        name: "bittensor",
        fetch: &[&["{python}", "-m", "bulk", "bittensor-fetch", "--out", "{work}/src"]],
        builds: &[
            Build {
                name: "subnets",
                argv: &["{python}", "-m", "bulk", "subnets", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/subnets-chain.json"],
                full_list: true,
            },
            Build {
                name: "subnet-sites",
                argv: &["{python}", "-m", "bulk", "subnet-sites", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/sites"],
                full_list: true,
            },
            Build {
                name: "subnet-github",
                argv: &["{python}", "-m", "bulk", "subnet-github", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/src/github"],
                full_list: true,
            },
        ],
        cwd: "",
        python: "",
        memory_bytes: 0,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "arxiv",
        fetch: &[&["{python}", "-m", "bulk", "arxiv-harvest", "--out", "{work}/oai", "--from", "{since}", "--until", "{until}"]],
        builds: &[Build {
            name: "arxiv",
            argv: &["{python}", "-m", "bulk", "arxiv", "--out", "{out}", "--as-of", "{as_of}", "--run", "{run}", "{work}/oai/*.xml"],
            full_list: false,
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
            argv: &["{python}", "-m", "crawler", "refresh", "edgar", "--since", "{since}", "--until", "{until}", "--out", "{out}", "--per-second", "1"],
            full_list: false,
        }],
        cwd: CRAWLER,
        python: CRAWLER_PYTHON,
        memory_bytes: CRAWLER_MEMORY,
        timeout_s: 6 * HOUR,
    },
    Source {
        name: "fred",
        fetch: &[],
        builds: &[Build { name: "fred", argv: &["{python}", "-m", "crawler", "refresh", "fred", "--out", "{out}"], full_list: false }],
        cwd: CRAWLER,
        python: CRAWLER_PYTHON,
        memory_bytes: CRAWLER_MEMORY,
        timeout_s: 80 * HOUR,
    },
];

pub fn find(name: &str) -> Option<&'static Source> {
    SOURCES.iter().find(|source| source.name == name)
}
