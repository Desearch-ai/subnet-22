# Why Desearch 2.0

Desearch 1.0 showed us that a network of miners and validators could help deliver useful search. It also showed us why good answers alone were not enough.

With 2.0, we want that work to build something lasting: a maintained collection of useful web information, an index for finding it quickly, and models that keep improving how it is used.

## What 1.0 achieved

In 1.0, miners supplied search results through their retrieval approaches, validators evaluated answers and the Desearch team operated the API.

That work produced useful results and gave us practical experience with customer-facing search and outside contributors. We are building on that contribution, not dismissing it.

## Where the approach fell short

**Good results could still take too long.** In operating 1.0, response time was a recurring limitation. An AI application needs relevant information soon enough to continue its task. Quality that arrives too late can make an otherwise useful service unsuitable.

**We depended on capabilities outside our control.** The 1.0 design evaluated answers without reliably distinguishing miner-operated retrieval from reliance on external search providers. A good response could therefore depend on another service's coverage, behavior and timing. The limitation was in the system we designed: useful answers did not necessarily give us capabilities we could retain and maintain.

**The work did not consistently leave enough reusable capability.** We needed retained source text, supporting passages and reproducible retrieval components that we could maintain and improve. Some capability accumulated, but access could still depend on an individual miner or provider continuing to supply it. This exposed a dependency we wanted to reduce.

The lesson was broader than “make the same request faster.” We needed more control over the information behind a response and more lasting value from the work used to produce it.

## Why build an index?

Think of an index as a prepared way to find information in a maintained collection. We collect useful pages, extract their content, remove duplicates and low-value material, organize what remains and prepare it for retrieval.

A customer query can then search information already prepared for use. Collection, cleaning and most document preparation happen in the background instead of being repeated on demand. Sources still need refreshing, and search still needs good models and ranking.

For Desearch, that creates three opportunities:

- **Faster retrieval:** move expensive preparation out of the ordinary query path.
- **More control over quality:** inspect source coverage, freshness and supporting text, and improve the models against the same maintained collection.
- **Reusable value:** use the underlying organized information across repeated searches and, where appropriate, selected data products.

An index does not automatically make search fast or accurate. Coverage, refresh policy, models, ranking and serving all matter. We must measure the resulting improvement. External indexes, licensed sources and hybrid approaches remain useful options; operating our own foundation does not require collecting every source ourselves.

## What changes in the design

| In 1.0                                                          | Intended change in 2.0                                                                     |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Miners supplied results for search requests                     | Contributors build and improve reusable data and retrieval components                      |
| Answer quality was the main object of assessment                | Assess submitted work, the resulting data quality and its effect on useful search          |
| Important capabilities could remain behind another service      | Retain permitted content and adopt reusable methods under explicit operating rights        |
| Customer responses depended on the participating retrieval path | Serve a tested index and model version while contribution work continues in the background |

These are design changes, not a claim that every miner used the same approach or that 2.0 is already deployed.

## What we learned

Accuracy, speed and dependable access belong together. Incentives should recognize work that builds useful capability, including data maintenance as well as better methods. The value of a collection depends on its coverage, quality and permitted uses, rather than its size alone.

The [company foundation](../company/company-foundation.md) sets out the wider ambition. Search is the first application of that maintained foundation; selected datasets and other AI uses can follow when there is a reason to build them.

Continue to [direction and first phase](02-direction-and-first-phase.md).
