from desearch_bot.categories import EXCLUDE, PRIORITY, UT1_LABELS, Catalogue


def _catalogue(**by_label):
    return Catalogue({label: set(hosts) for label, hosts in by_label.items()})


def test_every_excluded_label_is_one_we_can_actually_assign():
    assert EXCLUDE <= set(UT1_LABELS.values())


def test_priority_names_are_real_labels():
    assert set(PRIORITY) <= set(UT1_LABELS.values())


def test_labels_are_ordered_by_priority():
    catalogue = _catalogue(
        news=["example.com"], adult=["example.com"], blog=["example.com"]
    )
    assert catalogue.labels("example.com") == ["adult", "news", "blog"]


def test_unlisted_host_has_no_labels():
    assert _catalogue(adult=["porn.example"]).labels("news.example") == []


def test_excluded_reports_the_offending_label():
    catalogue = _catalogue(gambling=["casino.example"], news=["casino.example"])
    assert catalogue.excluded(catalogue.labels("casino.example")) == "gambling"


def test_a_domain_with_only_safe_labels_is_kept():
    catalogue = _catalogue(news=["paper.example"], sports=["paper.example"])
    assert catalogue.excluded(catalogue.labels("paper.example")) is None


def test_matching_is_on_the_exact_host_not_the_parent():
    """UT1 lists subdomains, so a bad subdomain must not condemn its registrable domain."""
    catalogue = _catalogue(adult=["nsfw.blogplatform.example"])
    assert catalogue.labels("blogplatform.example") == []
    assert catalogue.labels("nsfw.blogplatform.example") == ["adult"]


def test_casinos_and_porn_are_excluded_but_news_and_shopping_are_not():
    assert {"gambling", "adult"} <= EXCLUDE
    assert not {"news", "shopping", "blog", "education", "sports"} & EXCLUDE
