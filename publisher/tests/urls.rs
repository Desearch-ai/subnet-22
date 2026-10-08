//! URLs canonicalised and keyed as Python's `app.canonical` does; the expected values come from running it.

use desearch::canonical::{canonicalize, domain_of, unquote_plus};
use publisher::records::page_key;

const VECTORS: [(&str, &str, &str, &str); 18] = [
    (
        "https://www.Example.com/news/story?utm_source=x&id=7#top",
        "https://www.example.com/news/story?id=7",
        "example.com",
        "pages/example.com/02722b06dd321d090d4c8b6b2b099dcd30a7a544",
    ),
    (
        "https://example.com/search?q=hello%20world&flag",
        "https://example.com/search?q=hello+world&flag=",
        "example.com",
        "pages/example.com/a55572d147cbf733a5fffc29f071e64ffbc38a90",
    ),
    ("https://example.com/x?x=1&amp;y=2", "https://example.com/x?x=1&amp%3By=2", "example.com", "pages/example.com/fd3c57f6506740411816418f294299e725cd1b51"),
    (
        "  https://EXAMPLE.com/a?Ref=abc&FBCLID=1&ref%0A=2&utm_medium=&b=%E2%82%AC+%26\u{3000}",
        "https://example.com/a?b=%E2%82%AC+%26",
        "example.com",
        "pages/example.com/9a7e2d65ebdd10a57e2f075135f8e50eafa3d09e",
    ),
    (
        "https://example.com/a?copy=1&not=2&amp=3&region=en",
        "https://example.com/a?copy=1&not=2&amp=3&region=en",
        "example.com",
        "pages/example.com/9a21b0586736a6446e083e9efedc496bf331bd5f",
    ),
    (
        "https://M\u{fc}nchen.de/stra\u{df}e?q=\u{fc}&r=\u{20ac}",
        "https://m\u{fc}nchen.de/stra\u{df}e?q=%C3%BC&r=%E2%82%AC",
        "m\u{fc}nchen.de",
        "pages/m\u{fc}nchen.de/bfb0e9a1da811156b22fbe04fbb89f6c7c37a991",
    ),
    (
        "https://ex.com/p?a=%C3&b=%ZZ&c=%e2%82&d=%f0%9f%98",
        "https://ex.com/p?a=%EF%BF%BD&b=%25ZZ&c=%EF%BF%BD&d=%EF%BF%BD",
        "ex.com",
        "pages/ex.com/2676b23e39ab68b68d7f2d15b946370f83c68223",
    ),
    (
        "https://\u{130}STANBUL.com/x",
        "https://i\u{307}stanbul.com/x",
        "i\u{307}stanbul.com",
        "pages/i\u{307}stanbul.com/5711056500cd4431cb7f0cd6dda5663ef79c4fcf",
    ),
    ("HTTPS://Ex.com:443/p;params?x=1", "https://ex.com:443/p;params?x=1", "ex.com:443", "pages/ex.com:443/7c17edf2f440ab99ad5818bee887817acb7cb8d7"),
    (
        "https://ex.com/a&lt;b&gt;?fbcl\u{130}d=1&mc_cid=2&ito=3&cmpid=4&gclid=5&utm_=6&keep=7",
        "https://ex.com/a&lt;b&gt;?keep=7",
        "ex.com",
        "pages/ex.com/cd33931b9dc5d6ac4f62625ab0a77611f3689cab",
    ),
    ("//ex.com/no-scheme?z=1", "//ex.com/no-scheme?z=1", "ex.com", "pages/ex.com/c17db06b8f42f661a695a5349f896b2c8b9875bd"),
    ("https://ex.com/p?&&a&b=&=c&d=e=f", "https://ex.com/p?a=&b=&=c&d=e%3Df", "ex.com", "pages/ex.com/4f451ae36213c2941e6b0454a8f242554e6c3298"),
    ("https://ex.com/a?b=1#frag?x", "https://ex.com/a?b=1", "ex.com", "pages/ex.com/8128223e70895cc4d5cd1780f7f9e92d8e980bf6"),
    ("https://ex.com/&#x26;&#38;", "https://ex.com/&", "ex.com", "pages/ex.com/51b6ede86df2d12fbb7be44752a3206db7679272"),
    ("https://[::1]:8080/x", "https://[::1]:8080/x", "[::1]:8080", "pages/[::1]:8080/ad24e2dabf624bdfe45d5bbd609e7bc9e68ee750"),
    (
        "https://ex.com/path with space?q=a b+c&s=~-._*/",
        "https://ex.com/path with space?q=a+b+c&s=~-._%2A%2F",
        "ex.com",
        "pages/ex.com/866d7483bb18d85c1e8c35b0cc0665dbebf5a42b",
    ),
    ("mailto:someone@example.com", "mailto:someone@example.com", "", "pages//5a9db2ee430912e7250da417e3a5554a47f79845"),
    (
        "https://ex.com/\u{e9}t\u{e9}?x=\u{e9}%C3%A9",
        "https://ex.com/\u{e9}t\u{e9}?x=%C3%A9%C3%A9",
        "ex.com",
        "pages/ex.com/4a407b17d5876eb32ef747b5a1081da4e60f888c",
    ),
];

#[test]
fn urls_canonicalise_and_key_as_python_does() {
    for (raw, canonical, domain, key) in VECTORS {
        let got = canonicalize(raw).unwrap();
        assert_eq!(got, canonical, "{raw:?}");
        assert_eq!(domain_of(&got).unwrap(), domain, "{raw:?}");
        assert_eq!(page_key(&got).unwrap(), key, "{raw:?}");
    }
}

#[test]
fn urls_python_rejects_are_rejected() {
    assert!(canonicalize("https://ex.com/[x").is_ok());
    for raw in ["https://a]b.com/", "https://[1.2.3.4]/", "https://ex\u{2100}.com/"] {
        assert!(canonicalize(raw).is_err(), "{raw:?}");
    }
}

#[test]
fn bad_escapes_decode_as_python_does() {
    assert_eq!(unquote_plus("%E2%82A%f0%9f%98%80+%ed%a0%80%C0%AF%"), "\u{fffd}A\u{1f600} \u{fffd}\u{fffd}\u{fffd}\u{fffd}\u{fffd}%");
    assert_eq!(unquote_plus("caf\u{e9}%C3%A9%C3\u{e9}"), "caf\u{e9}\u{e9}\u{fffd}\u{e9}");
}
