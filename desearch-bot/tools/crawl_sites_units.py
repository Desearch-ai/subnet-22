"""Freeze the Python crawler's answers for page markup, URLs, robots.txt, sitemaps, charsets and wire URLs: `python crawl_sites_units.py ENGINE OUT.json`."""

from __future__ import annotations

import base64
import gzip
import json
import sys

HTML = [
    "",
    "  ",
    "\x0c",
    "\xa0",
    "<!-- x -->",
    "<!DOCTYPE html>",
    "</p>",
    "</div></html>",
    "<?php x ?>",
    "hello",
    "&amp;",
    "​",
    "\x00",
    "<!-- c -->\xa0",
    "<?xml version='1.0' encoding='utf-8'?><html><a href=x>",
    "<?xml  encoding='x'?><a href=y>",
    "<?xml encoding='x'?><a href=y>",
    "<?XML version='1.0' encoding='utf-8'?><a href=x>",
    " <?xml version='1.0' encoding='utf-8'?><a href=x>",
    "<?xml version='1.0'?><html lang=fr>",
    "<html lang=en><html lang=fr dir=x>",
    "<p>x</p><html lang=fr>",
    "hello<html lang=fr>",
    "<!-- c --><html lang=fr>",
    "﻿<html lang=en>",
    "<html lang=' EN-us '>",
    "<HTML LANG=fr>",
    '<html LANG="en" lang=fr>',
    "<html xml:lang='fr'>",
    "<head><html lang=fr>",
    "</p><html lang=fr>",
    "<script><a href=s></script><a href=1>",
    "<script>if (a<b) { x='</scr'+'ipt>'; }</script><a href=2>",
    "<script><!-- <script> </script> --> <a href=esc></script><a href=after>",
    "<noscript><a href=ns></noscript>",
    "<style><a href=st></style>",
    "<title><a href=t></title><a href=u>",
    "<textarea><a href=ta></textarea>",
    "<xmp><a href=x></xmp>",
    "<iframe><a href=if></iframe>",
    "<noembed><a href=ne></noembed>",
    "<noframes><a href=nf></noframes>",
    "<template><a href=tp></template>",
    "<plaintext><a href=pt>",
    "<script/><a href=sc>",
    "<svg><style><a href=svgstyle></style></svg><a href=z>",
    "<math><a href=m></math>",
    "<a href='a&amp;b&nbsp;c&notit;d&#x41;&#0;&ampx&lt'>",
    "<a href=a&amp>",
    "<a href=&#x110000;>",
    "<a href=&#128;>",
    "<a href='&#x80;&euro;&Eacute;&eacute'>",
    "<a href='&AMP;&LT;&copy&copyx&not;&notin;&semi;'>",
    '<a href="x\0y">',
    "<a href=x>\x00",
    "<a HREF=X REL=NoFollow href=y>",
    "<a/href=z>",
    "<a href=1><a href=2></a></a>",
    "<![CDATA[<a href=cd>]]><a href=after>",
    "<!--><a href=c1>",
    "<!---><a href=c2>-->",
    "<! a><a href=b1>",
    "</a href=e>",
    "<base href=/b1><base href=/b2><a href=x>",
    "<base><base href=/b2><a href=x>",
    "<base href=''><a href=x>",
    "<meta name=ROBOTS content='NoIndex, x'><meta name=robots>",
    "<meta name=desearchbot content=nofollow>",
    "<meta content=none name=robots>",
    "<meta name='googlebot' content='noindex'>",
    "<a href=1></body><a href=2></html><a href=3>",
    "</html><a href=2>",
    "<svg></html></svg><a href=2>",
    "<script></html></script><a href=2>",
    "<a href=1></HTML ><a href=2>",
    "x</html><a href=2>",
    "<table><a href=t1><tr><a href=t2>",
    "<select><a href=s1></select>",
    "<a href=1 <a href=2>",
    "<a href='1'href='2'>",
    '<a href="1"/href=2>',
    "<a\nhref\n=\n'nl'\n>",
    "<a href='\n  /spaced  \n'>",
    '<a href="javascript:void(0)">',
    '<a href="#top">',
    "<a href='//other.test/x'>",
    "<a href='?q=1'>",
    "<a href='../up'>",
    "<a href='http://[broken'>",
    "<a href>",
    "<a href=''>",
    "<frameset><a href=fs>",
    "<image href=i><a href=im>",
    "<a rel='a nofollow' href=r1><a rel='NOFOLLOW\tx' href=r2><a rel=nofollowx href=r3><a rel=' nofollow' href=r4>",
    "<p>\r\n<a href='cr\r\nlf'>",
    "<a href='tab\there'>",
    '<a href="été">',
    '<a href="/caf%C3%A9">',
    "<html><head><base href='https://cdn.test/root/'></head><body><a href=rel>",
    "<html><body><a href=x></body></html>\n<!-- end -->",
]
URLS = [
    ("", "https://ex.test/a/b"),
    ("/a/b?utm_source=x&id=2#top", "https://Ex.test:443/"),
    ("mailto:x@ex.test", ""),
    ("http://ex.test:8080/", ""),
    ("https://a.test: Managing Supply/x", ""),
    ("https://a.test:99999/", ""),
    ("https://a.test:8080/x#y", ""),
    ("https://a.test:0/x", ""),
    (" https://u:p@A.test/x?b=1+2&a=%41&e&ref=z&REF=y&gclid=1&q=a%2Fb ", ""),
    ("?x=é", "https://a.test/p"),
    ("https://a.test", ""),
    (
        "https://a.test/x?utm_=1&utm_medium=2&UTM_X=3&fbclid=4&mc_cid=5&mc_eid=6&cmpid=7&ito=8&ref=9&refx=10&xref=11",
        "",
    ),
    ("https://a.test/x?a=1&&b=2&=3&c==4&d=%zz&e=%E2%82%AC&f=%ff&g=a+b%20c", ""),
    ("https://a.test/x?;a=1;b=2", ""),
    ("https://a.test/x?", ""),
    ("https://a.test/x?#frag", ""),
    ("//b.test/y", "https://a.test/x"),
    ("../../z", "https://a.test/a/b/c"),
    ("./", "https://a.test/a/b"),
    ("ftp://a.test/x", ""),
    ("javascript:alert(1)", "https://a.test/"),
    ("https://[::1]:8443/x", ""),
    ("https://[::1/x", ""),
    ("http://a.test:80/x", ""),
    ("https://A.TEST./x", ""),
    ("https://éxample.test/x", ""),
    ("https://a.test/été x", ""),
    ("https:/a.test/x", ""),
    ("https:x", "https://a.test/b/c"),
    ("https://a.test\\x", ""),
    ("HTTPS://a.test/X", ""),
    ("\thttps://a.test/x\n", ""),
    ("https://a.test/x?q=%2B%20+", ""),
    ("https://a.test/x?ref\n=1", ""),
    ("https://a.test/x?ſef=1", ""),
    ("https://a.test/x?q=€", ""),
    ("https://user@a.test:443/x", ""),
    ("https://a.test:/x", ""),
    ("https://a.test:+80/x", ""),
    ("x;", "https://e.test/p/q"),
    ("x;?a=1", "https://e.test/p/q"),
    ("x;p", "https://e.test/p/q"),
    (";p", "https://e.test/p/q"),
    ("?q=1", "https://e.test/p/q;b"),
    ("..;p", "https://e.test/p/q/r"),
    ("https://e.test/a;", "https://e.test/"),
    ("#f", "https://e.test/p;x?y=1"),
    ("a/b;c/d;e", "https://e.test/p/"),
    ("", ""),
]
SCOPES = [
    ("https://ex.test/doc/a", "ex.test", ["/doc/"]),
    ("https://ex.test/blog/a", "ex.test", ["/doc/"]),
    ("https://ex.test/a.PDF", "ex.test", []),
    ("https://other.test/a", "ex.test", []),
    ("https://EX.test/a", "ex.test", []),
    ("https://ex.test/a.json?x=1", "ex.test", []),
    ("https://ex.test/a.jsonx", "ex.test", []),
    ("https://ex.test:8080/a", "ex.test", []),
    ("https://ex.test/a.tar.gz", "ex.test", []),
    ("https://ex.test/feed.rss", "ex.test", []),
]
ROBOTS = [
    "User-agent: *\nDisallow: /private/\nCrawl-delay: 3\nSitemap: https://a.test/sm.xml\n",
    "User-agent: DesearchBot\nDisallow: /\n\nUser-agent: *\nAllow: /\n",
    "User-agent: *\nDisallow: /*?tab=\n",
    "User-agent: desearch\nDisallow: /a\n\nUser-agent: *\nDisallow: /b\n",
    "User-agent: bot\nDisallow: /a\n",
    "User-agent: desearchbotx\nDisallow: /a\n",
    "User-agent: x-desearchbot\nDisallow: /a\n",
    "User-agent: Desearch*\nDisallow: /a\n",
    "useragent: *\ndissallow: /typo\nDisallow: /$\nAllow: /docs/index.html\nDisallow: /docs/\n",
    "User-agent: *\nDisallow: /caf%C3%A9\nDisallow: /a b\nDisallow: /*.pdf$\nDisallow: /%7euser\n",
    "Disallow: /orphan\nUser-agent: *\nAllow: /\nUser-agent: other\nDisallow: /\n",
    "User-agent: *\nUser-agent: other\nDisallow: /\n",
    "Sitemap: /s.xml\nsite-map: https://x/t.xml\nsitemaps:https://x/u.xml\n",
    "User-agent: *\nCrawl-delay: 1_5\n",
    "User-agent: *\nCrawl-delay: inf\n",
    "User-agent: *\nCrawl-delay: -1\n",
    "User-agent: *\nCrawl-delay: 0\n",
    "User-agent: *\nCrawl-delay: 2.5 # comment\n",
    "User-agent: DesearchBot\nCrawl-delay: 7\nUser-agent: *\nCrawl-delay: 1\n",
    "User-agent: *\nDisallow: /a\nUser-agent: DesearchBot\nAllow: /a/b\n",
    "User-agent: *\nAllow: /a/b\nDisallow: /a\n",
    "User-agent: *\nDisallow: /a\nAllow: /a\n",
    "User-agent: *\nDisallow: /*/print*\nDisallow: /*.php$\nAllow: /*.php$\n",
    "User-agent: *\nDisallow: /x$y\n",
    "User-agent: *\nDisallow: *\n",
    "User-agent: *\nDisallow:\n",
    "User-agent: *\r\nDisallow: /cr\r\n",
    "﻿User-agent: *\nDisallow: /bom\n",
    "User-agent * \nDisallow /nocolon\n",
    "User agent: *\nCrawl delay: 4\nDisallow: /spaced\n",
    "user-agent: *\ndisallow: /UPPER\n",
    "User-agent: *\nDisallow: /search?\nDisallow: /q=\n",
    "User-agent: *\nDisallow: http://a.test/full\n",
    "User-agent: *\nNoindex: /x\nDisallow: /y\nHost: a.test\n",
    "garbage line\nUser-agent: *\nDisallow: /g\n",
]
ROBOT_URLS = [
    "https://a.test/",
    "https://a.test/page",
    "https://a.test/private/x",
    "https://a.test/x?tab=1",
    "https://a.test/x",
    "https://a.test/a",
    "https://a.test/a/b",
    "https://a.test/b",
    "https://a.test/typo",
    "https://a.test/docs/",
    "https://a.test/docs/index.html",
    "https://a.test/docs/a",
    "https://a.test/café/menu",
    "https://a.test/caf%c3%a9",
    "https://a.test/a%20b",
    "https://a.test/a b",
    "https://a.test/f/x.pdf",
    "https://a.test/f/x.pdf?y",
    "https://a.test/robots.txt",
    "https://a.test/robots.txt;x",
    "https://a.test/orphan",
    "https://a.test/2024/print/x",
    "https://a.test/x.php",
    "https://a.test/x$y",
    "https://a.test/x%24y",
    "https://a.test/cr",
    "https://a.test/bom",
    "https://a.test/nocolon",
    "https://a.test/spaced",
    "https://a.test/UPPER",
    "https://a.test/upper",
    "https://a.test/search?q=1",
    "https://a.test/search",
    "https://a.test/q=1",
    "https://a.test/full",
    "https://a.test/http://a.test/full",
    "https://a.test/y",
    "https://a.test/g",
    "https://a.test/~user",
    "https://a.test/%7Euser",
    "https://a.test//double",
    "https://a.test/a#frag",
    "https://a.test/a;p",
]
SITEMAP_NS = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
SITEMAPS = [
    f'<?xml version="1.0"?><urlset {SITEMAP_NS}><url><loc> https://ex.test/a </loc></url><url><loc>https://ex.test/b?x=1&amp;y=2</loc></url></urlset>',
    f"<sitemapindex {SITEMAP_NS}><sitemap><loc>https://ex.test/s1.xml.gz</loc></sitemap></sitemapindex>",
    "not xml",
    "",
    "<urlset><url><loc>https://ex.test/a?x&y</loc></url></urlset>",
    "<urlset><loc>a</loc>",
    "<urlset></urlset><x/>",
    "<urlset><loc>a</lc></urlset>",
    '<urlset xmlns:i="urn:img"><url><loc>a<!-- c -->b</loc><i:loc>img</i:loc><loc><![CDATA[c]]>d</loc></url></urlset>',
    '<urlset xmlns="urn:other"><url><loc>other-ns</loc></url></urlset>',
    "<urlset><url><loc>&#x41;&#66;&lt;&gt;&quot;&apos;</loc></url></urlset>",
    "<urlset><url><loc>&nbsp;x</loc></url></urlset>",
    "<urlset><url><loc>&#0;x</loc></url></urlset>",
    "  \n<?xml version='1.0'?><urlset><loc>ws</loc></urlset>",
    "<?xml version='1.0'?>\n<urlset>\n<loc>nl</loc>\n</urlset>\n",
    "<urlset><loc>a</loc></urlset>trailing",
    "<urlset><loc>a</loc></urlset><!-- ok -->",
    "<mysitemapindex><loc>idx</loc></mysitemapindex>",
    "<urlset><url><loc/><loc>  </loc><loc>x<b>y</b>z</loc></url></urlset>",
    "<urlset><url a='1' a='2'><loc>dup</loc></url></urlset>",
    "<urlset><url a=1><loc>unquoted</loc></url></urlset>",
    "<urlset><url a='x&y'><loc>attr</loc></url></urlset>",
    "<?xml version='1.0' encoding='ISO-8859-1'?><urlset><loc>caf\xe9</loc></urlset>",
    "<urlset><loc>caf\xc3\xa9</loc></urlset>",
    "<urlset><loc>bad\xff</loc></urlset>",
    "<!DOCTYPE urlset><urlset><loc>dt</loc></urlset>",
    "<urlset><loc>a</loc><loc>b</loc></urlset>",
    "<s:urlset xmlns:s='http://www.sitemaps.org/schemas/sitemap/0.9'><s:loc>pre</s:loc></s:urlset>",
]
CHARSETS = [
    (b"caf\xe9", "text/html; charset=ISO-8859-1"),
    (b"\x80\x81\x8d\x9d", "text/html; charset=windows-1252"),
    (b"\x80\x81", "text/html; charset=cp1252"),
    (b"a\xe9", "text/html; charset=us-ascii"),
    (b"caf\xc3\xa9", "text/html; charset=no-such"),
    (b"caf\xc3\xa9\xff", "text/html"),
    (b"ok \xe2\x82 end", "text/html; charset=utf-8"),
    (b"\xed\xa0\x80x\xf4\x90\x80\x80y\xc0\xafz", "text/html; charset=UTF8"),
    (b"\x81\x83\x88", "text/html; charset=windows-1250"),
    (b"\xd0\xdd\xfe", "text/html; charset=iso-8859-9"),
    (b"\xa4", "text/html; charset=iso-8859-15"),
    (b"\x82\xa0", "text/html; charset=shift_jis"),
    (b"\xc4\xe3", "text/html; charset=gbk"),
    (b"\xa1\xa1", "text/html; charset=euc-kr"),
    (b"\xc1", "text/html; charset=koi8-r"),
    (b"x", 'text/html; charset="utf-8"'),
    (b"\xe9", "text/html; charset='latin1'"),
    (b"\xe9", "text/html; CHARSET=Latin-1"),
    (b"\xe9", "text/html;charset=utf_8"),
]
WIRE = [
    "https://a.test/authorize?back=%2Fart",
    "https://a.test/A%20b/%41%2f%2B%3a/café/x y?q=a%2Bb%3D&s=%7e+1&t=x y",
    "https://a.test/a/./b/../c%zz%4",
    "https://a.test",
    "https://a.test/x'y?a='b'&c=[d]",
    "https://a.test/%7Euser/%3B;/%40@?a=%26&b=%3F?&c=%23",
    'https://a.test/{x}|y^`z"<>',
    "https://a.test/x%2e%2e/y",
    "https://a.test/a/../../b",
    "https://a.test/%",
    "https://a.test/a%2",
    "https://a.test/?%",
    "https://a.test/€?€=€",
]


def main() -> int:
    engine, out = sys.argv[1:3]
    sys.path.insert(0, engine)
    from yarl import URL

    from crawler.links import (
        in_scope,
        normalize,
        page_links,
        parse_html,
        robots_meta,
        sitemap_entries,
    )
    from crawler.net import Response
    from crawler.robots import parse
    from crawler.scope import declared_lang

    def html_case(text: str):
        try:
            doc = parse_html(text)
        except Exception as e:  # noqa: BLE001
            return {"error": type(e).__name__}
        if doc is None:
            return None
        try:
            links = page_links(doc, "https://ex.test/p/q")
        except ValueError:
            links = "ValueError"
        return {
            "lang": declared_lang(doc),
            "meta": sorted(robots_meta(doc)),
            "links": links,
        }

    def sitemap_case(body: bytes):
        try:
            return list(sitemap_entries(body))
        except Exception as e:  # noqa: BLE001
            return {"error": type(e).__name__}

    def robots_case(body: str):
        r = parse("a.test", 200, body)
        return {
            "delay": r.delay,
            "sitemaps": r.sitemaps,
            "allowed": [u for u in ROBOT_URLS if r.allowed(u)],
        }

    bodies = [s.encode("latin-1") for s in SITEMAPS]
    bodies.append(gzip.compress(bodies[1], mtime=0))
    bodies.append(b"\x1f\x8bnot gzip")
    bodies.append(gzip.compress(bodies[0], mtime=0)[:-12])
    result = {
        "html": [[t, html_case(t)] for t in HTML],
        "normalize": [[u, b, normalize(u, b)] for u, b in URLS],
        "scope": [[u, h, p, in_scope(u, h, tuple(p))] for u, h, p in SCOPES],
        "robots": [[t, robots_case(t)] for t in ROBOTS],
        "robot_urls": ROBOT_URLS,
        "sitemaps": [[base64.b64encode(b).decode(), sitemap_case(b)] for b in bodies],
        "charsets": [
            [
                base64.b64encode(b).decode(),
                ct,
                Response("u", 200, {"content-type": ct}, b).text(),
            ]
            for b, ct in CHARSETS
        ],
        "wire": [
            [u, (lambda x: f"{x.scheme}://{x.raw_host}{x.raw_path_qs}")(URL(u))]
            for u in WIRE
        ],
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
