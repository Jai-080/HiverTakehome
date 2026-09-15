# Failure Analysis

Five real failure modes pulled from `surface_failure_candidates.py`'s output against the 200-row golden set, each with a fixable-or-fundamental call.

## 1. Refund/return language systematically misread as delivery or billing
**Fixable — root cause of #2 below.**

All 5 top misclassifications are the same confusion pair: `refund_return_dispute` mistaken for `delivery_shipping_issue` or `billing_subscription_dispute`, every one at `predicted_confidence=0.95`. Example: tweet 869858, "I haven't got my cash back. I purchased prime membership & added money to wallet" — predicted `billing_subscription_dispute`, true `refund_return_dispute`. The classifier is pattern-matching surface vocabulary (payment/delivery terms) instead of the actual ask (money back). Fixable the same way the earlier billing-vs-delivery confusion was: add few-shot examples in the classifier prompt explicitly distinguishing "wants a refund" from "mentions a delivery or payment method." The fact that confidence sits at 0.95 on every wrong call here is itself worth flagging as evidence the confidence score isn't well-calibrated to actual correctness.

## 2. That confusion cascades into missed escalations
**Fixable only by fixing #1 — not a separate bug in the escalation policy itself.**

Cross-referencing: tweets 1993267, 468225, and 206777 appear on both the misclassification list and the missed-escalation list. `refund_return_dispute` is a high-risk always-escalate intent, but these tweets never got labeled that way, so the escalation gate never triggered. The policy logic is sound in isolation — the failure is entirely upstream. Worth stating explicitly: a safety-critical gate is only as strong as the classifier feeding it, and this is a concrete example of that dependency failing silently.

## 3. Retrieval grounds tone as well as content, and gets it backwards on bad news
**Partially fixable, currently a fundamental limitation of pure similarity retrieval.**

Tweet 1273380: customer reports a product defect ("white dot on top," asks Amazon to look into it); the drafted reply says "We're thrilled to hear about your new OnePlus 5... hope it brings you joy" — a full sentiment inversion, scoring the worst possible 1.0/1.0 on groundedness and hallucination. The retrieval step matched on lexical similarity to a positive unboxing reply without checking that its polarity matched the customer's actual complaint. A sentiment/polarity check between the incoming tweet and retrieved candidates before drafting could mitigate this, but pure similarity retrieval has no such signal built in today — this is an architecture limitation, not a bug.

## 4. Near-verbatim copying persists at real magnitude despite the earlier fix
**A known, inherent tension in RAG-style grounding, not fully fixable without a quality tradeoff.**

Tweet 2835636, 92% token overlap on a price-match question — the reply essentially recites the retrieved historical example. Tightening the no-copy instruction further risks the model drifting from the grounded example entirely, reintroducing the hallucination risk the retrieval was meant to prevent — this is a real tradeoff between grounding and originality, not a simple bug to patch.

## 5. The name-ban safety heuristic false-positives on capitalized common nouns
**Fixable, low priority/cosmetic.**

Tweet 2152750, "A big thank you for delivering the product on Diwali day" — the reply was flagged for supposedly leaking the name "Diwali," which is a festival, not a person. Trivial to fix (expand the allowlist) but functionally harmless today since the flag is informational only — worth one sentence in the report as evidence of the tradeoff made when banning all capitalized words categorically for safety.
