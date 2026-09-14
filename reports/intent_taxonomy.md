# AmazonHelp Intent Taxonomy

Derived from two independent clustering runs, both at k=8:

1. Gemini `gemini-embedding-001`, 600-sample, pre-PII/first-touch fixes.
2. Local `BAAI/bge-large-en-v1.5`, 1500-sample, post-PII/first-touch fixes
   (current `reports/intent_clusters_raw.md`).

Both runs had silhouette scores in the near-zero range (0.01-0.04, see
`DECISIONS.md`) — short informal support tweets don't form geometrically
separated clusters in embedding space. This taxonomy was finalized by a full
manual read of both raw exports, not by trusting the mechanical cluster
boundaries. 8 classifier intents, plus one escalation override that sits
outside the intent classifier entirely.

---

## 1. Delivery/Shipping Issue

Late, damaged, missing, or wrong item; or a tracking status ("delivered")
that doesn't match what the customer actually received.

**Note:** clustering split this into three sub-flavors that are merged here
for classification purposes but are worth separating in failure analysis —
(a) damage/wrong-item, (b) Prime shipping-speed delay, (c) status/tracking
confusion (app says "delivered", customer says otherwise).

**Examples:**
- An iron box bought from Amazon was a damaged item because of less precautions in packing. So Replaced but no pickup. #Disappointed
- Several parcels delayed lately despite choosing next day and being a prime customer. What's going on?
- I'm confused where my delivery is. App says "delivered", handed to resident yesterday - with email after saying failed delivery attempt as no one in. Second attempt today with email saying couldn't locate my delivery address?
- dont even need to open the box to hear out lovely new glasses are completely smashed!!!!
- Here we go again. My package delivered to someone else. Happens approx. every other time I order which is a lot! They don't care.

## 2. Refund/Return & Dispute Resolution

Refund requests, balance/money disputes, subscribe-and-save issues.

**Examples:**
- will I be getting my postage refunded?
- A order placed by mistakenly. I cancelled the order. My amazon pay balance showing 0. My balance was 1299. When the amount will show on my balance?
- very poor experience with subscribe and save. Order never delivered. Refund messed up. Very poor show.
- Pls help received a damaged product. . order #<PHONE>.

## 3. Account & Access Issues

Login lockout, password reset failure, checkout/exchange technical friction.

**Note:** shares vocabulary with #2 in raw clustering (both use
"account"/"help"/"order") — kept as a distinct classifier intent despite
that, since the two need different handling (access recovery vs. money
movement).

**Examples:**
- Hey you locked me out of my account after I bought something and now won't let me log back in, what gives?!?
- sorry let me say that again in english, I cannot log in into my account even after reseting the password multiple times (i even got the captcha), please help I wanted to buy something to get the 1st
- not able to check out with exchange despite the product page saying so. Help
- you do not cater exchange offer in our pincode. That's bad. We are not able to shop with your site

## 4. Billing/Subscription Charge Dispute

Unwanted Prime membership charges, trial charges, cancellation difficulty.

**Examples:**
- second time in as many months that Amazon has gone into my account and stolen money for a Prime membership I do not want.
- how to stop my prime membership, since i was scammed and you do nothing to me, i need to cancel my membership.
- I activated the Amazon Trial prime and £ 1 was deducted from my Visa card !
- Hi my account is meant to be closed but im still getting emails about prime payment issues?

## 5. Digital Service/Device Technical Support

Echo, Kindle, Alexa, app bugs, cross-device sync issues.

**Examples:**
- Just as an aside, what the hell have done to the Kindle App?! It's bloody awful, give me back my carousel!!
- Since the most recent update my spot in my books will no longer sync between my devices. All settings are correct, so that isn't the problem. Is anyone else having this problem?
- Bought Minecraft from Amazon Store on previous device. Wiped & sold device, Xferred acct to new device, now I get this - help!
- still waiting for my #echo invite.. been 2 weeks

## 6. Marketplace/Seller Trust

Counterfeit products, fake/misleading listings.

**Lower confidence:** clear in the first (Gemini-embedding) clustering run,
not clearly reproduced in the second (bge-large) run's 1500-sample — none of
the 8 clusters in the current `intent_clusters_raw.md` isolate this topic
cleanly. Kept as a distinct classifier intent anyway because it's
operationally distinct (routes to seller trust/safety, not the standard
refund flow) even though the clustering evidence for it is mixed. Only two
unambiguous examples were found; a third is included as a weaker, topically
adjacent case rather than padded out to hide the thin evidence.

**Examples** (from the first clustering run):
- #PehleKaroPuriTayyari #FakeListings available at 699 without any deal, Lightning deal at 799. Same product with different MRP.
- Why does sell fake Maybelline products and even after many complaints, refuse to remove it and suspend the seller?
- Now tell me, how come amazon allowing having 2 account using same email id? Confusing for consumer. Such glitches can hamper consumer faith *(weaker fit — platform-trust-adjacent, not a clean counterfeit-product case)*

## 7. Positive Feedback / No Action Needed

Genuine gratitude or praise, requiring no support action.

**Explicit caveat:** watch for sarcasm — lexical positivity is not
sufficient. "Thanks for leaving my parcel outside in the rain - very
clever!!!" scores as "positive" on any keyword-based sentiment rule but is
actually a complaint. This intent must be LLM-classified with sarcasm-aware
few-shot examples, not a keyword/lexicon rule.

**Examples:**
- So my will be arriving before 31st of October Thanks guys for speedy shipping.
- I love online shopping. Christmas shopping is done. Thanks !
- I just did the biggest shop of my life..... 😂😂
- Thanks for leaving my parcel outside in the rain - very clever!!! *(sarcasm trap — NOT actually positive, included here deliberately as the counterexample)*

## 8. General Complaint / Other

Venting or off-topic requests with no specific actionable resolution path.

**Examples:**
- Amazon's customer service is very nonsense its customer care service has become very nonsense, not helping people pockets are being stupid
- I got fed up with stubborn attitude. It seems Amazon team supports fraud.
- I was looking forward to my Amazon order all day. But some asshole stole it mid transit. Fuck you and I hope karma gets you back.
- I'm almost positive and the have it out for me.

---

## Escalation override: Security/Fraud Alert

Not a classifier intent — too rare to reliably learn from clustering (no
cluster in either run isolates it; these are scattered singletons across
several clusters). Instead, this is a deterministic escalation-policy
override: regardless of which of the 8 intents above a message is
classified into, specific fraud/security signals force escalation rather
than auto-handling.

**Trigger examples, pulled from real data:**
- "Hi Where can I forward an email I received from Amazon that is suspicious, it saying my account is locked which it isn't." *(phishing forward)*
- "how to stop my prime membership, since i was scammed and you do nothing to me..." *(scam claim)*
- "No. You are not listening. Account says all items delivered. Pack was opened. This is fraud. Police informed." *(fraud + law enforcement already involved)*
- "Hi, you probably also want to know that just DMed me pretending to be you and tried to get my CC details…" *(impersonation / credential phishing attempt)*
