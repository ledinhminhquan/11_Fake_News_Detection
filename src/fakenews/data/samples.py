"""Built-in seed dataset (offline fallback + tests).

A small, balanced set of original/synthetic **labeled news samples** (real vs
fake), a small **evidence corpus**, and a few **claims with gold verdicts** — so
that with NO torch and NO network:

* the TF-IDF + LogReg classifier trains + predicts (sklearn only);
* the agentic fact-check retrieves evidence + judges stance (lexical fallback);
* classification + fact-check metrics compute against the gold;
* the agent FSM completes end-to-end producing a verdict with citations.

The texts are original/synthetic (no copied dataset) and deliberately avoid real
named individuals; "fake" items are clearly false/sensational claims and "real"
items are sober factual statements, so the linear baseline can separate them. On
Colab a real dataset (LIAR / a clean fake-news corpus) + FEVER evidence are used.
"""

from __future__ import annotations

from typing import Dict, List

# label: 0 = real (factual/sober), 1 = fake (false/sensational/misinformation)
SEED_NEWS: List[Dict] = [
    # ---- real (0) ----
    {"id": "r01", "title": "Health agency updates handwashing guidance", "label": 0,
     "text": "The national health agency reissued its recommendation that regular handwashing with soap reduces the transmission of common respiratory and gastrointestinal infections."},
    {"id": "r02", "title": "Study links exercise to lower heart-disease risk", "label": 0,
     "text": "A peer-reviewed study published in a cardiology journal reported that moderate weekly physical activity was associated with a measurable reduction in cardiovascular disease risk across the cohort."},
    {"id": "r03", "title": "City council approves new bus routes", "label": 0,
     "text": "The city council voted to fund three additional bus routes next year, citing ridership data and a transit study presented at the public meeting."},
    {"id": "r04", "title": "Researchers sequence genome of a deep-sea fish", "label": 0,
     "text": "Marine biologists published the complete genome of a deep-sea anglerfish, noting adaptations to low light and high pressure that may inform future ecological research."},
    {"id": "r05", "title": "Central bank holds interest rate steady", "label": 0,
     "text": "The central bank announced it would keep its benchmark interest rate unchanged this quarter, pointing to stable inflation figures and steady employment data."},
    {"id": "r06", "title": "New telescope captures distant galaxy", "label": 0,
     "text": "Astronomers released images from a space telescope showing a distant galaxy, providing data that researchers will use to study early star formation."},
    {"id": "r07", "title": "School district adds free meal program", "label": 0,
     "text": "The school district announced a program to provide free breakfast to elementary students, funded by a state grant and beginning next semester."},
    {"id": "r08", "title": "Engineers test earthquake-resistant bridge design", "label": 0,
     "text": "Civil engineers reported results from a scale-model test of a bridge support designed to better absorb seismic forces, with findings submitted for peer review."},
    {"id": "r09", "title": "Vaccination campaign reduces measles cases", "label": 0,
     "text": "Public-health officials reported a decline in measles cases in a region following a routine childhood vaccination campaign, consistent with prior immunization data."},
    {"id": "r10", "title": "Company reports quarterly earnings", "label": 0,
     "text": "A technology company reported quarterly revenue in line with analyst expectations and announced modest growth in its cloud services division."},
    {"id": "r11", "title": "Weather service forecasts heavy rain", "label": 0,
     "text": "The meteorological service issued a forecast of heavy rainfall over the weekend and advised residents in low-lying areas to monitor local flood updates."},
    {"id": "r12", "title": "Library expands digital lending", "label": 0,
     "text": "The public library expanded its digital lending catalog, allowing cardholders to borrow more e-books and audiobooks through the existing mobile application."},
    {"id": "r13", "title": "Agricultural study examines drought-resistant crops", "label": 0,
     "text": "Agronomists published field-trial results comparing several drought-resistant crop varieties, reporting modest yield improvements under low-water conditions."},
    {"id": "r14", "title": "Hospital opens new wing", "label": 0,
     "text": "A regional hospital opened a new wing to expand outpatient capacity, funded through a combination of public budget and private donations."},
    {"id": "r15", "title": "Recycling rates rise in pilot program", "label": 0,
     "text": "A municipal pilot program reported an increase in household recycling participation after introducing curbside collection and an information campaign."},
    {"id": "r16", "title": "Researchers map coral reef recovery", "label": 0,
     "text": "Ecologists documented partial recovery of a coral reef over several years, attributing the change to reduced local pollution and stable water temperatures during the study period."},
    {"id": "r17", "title": "University launches scholarship fund", "label": 0,
     "text": "A university announced a scholarship fund for first-generation students, supported by alumni contributions and to be awarded based on financial need."},
    {"id": "r18", "title": "Transit authority tests electric buses", "label": 0,
     "text": "The transit authority began a trial of electric buses on two routes to evaluate range, maintenance costs, and passenger experience before a wider rollout."},
    {"id": "r19", "title": "Survey finds steady consumer spending", "label": 0,
     "text": "A monthly economic survey reported steady consumer spending, with small gains in services offset by flat retail sales, according to the agency's figures."},
    {"id": "r20", "title": "Researchers publish air-quality data", "label": 0,
     "text": "Environmental scientists released a year of air-quality measurements for a metropolitan area, noting seasonal variation and a long-term downward trend in particulate levels."},
    # ---- fake (1) ----
    {"id": "f01", "title": "SHOCKING: Drinking bleach cures every virus overnight, insiders reveal", "label": 1,
     "text": "Anonymous insiders claim that drinking household bleach completely cures all known viruses within hours, a miracle the authorities are supposedly hiding from the public."},
    {"id": "f02", "title": "Scientists PROVE the moon is a hologram projected by satellites", "label": 1,
     "text": "A viral post asserts that the moon is actually a giant hologram projected by secret satellites, and that every space agency in the world is covering it up."},
    {"id": "f03", "title": "Miracle pill melts 30 pounds in a single day, doctors furious", "label": 1,
     "text": "An online ad claims a single miracle pill can melt thirty pounds of fat in one day with no diet or exercise, and that doctors are furious it was leaked."},
    {"id": "f04", "title": "BREAKING: 5G towers secretly control the weather, whistleblower warns", "label": 1,
     "text": "A self-described whistleblower alleges that 5G cell towers are secretly used to control the weather and cause storms on demand, with no evidence provided."},
    {"id": "f05", "title": "Government admits vaccines contain mind-control microchips", "label": 1,
     "text": "A fabricated report claims the government secretly admitted that routine vaccines contain microchips designed to control people's thoughts, a claim with no factual basis."},
    {"id": "f06", "title": "EXPOSED: Drinking water turns people into zombies, scientists silenced", "label": 1,
     "text": "A sensational article claims that ordinary tap water is turning people into zombies and that the scientists who discovered this have all been silenced overnight."},
    {"id": "f07", "title": "Aliens landed downtown last night and bought the entire city, sources say", "label": 1,
     "text": "Unnamed sources claim that aliens landed in the city center last night and purchased the entire city using gold bars, an event no official has confirmed."},
    {"id": "f08", "title": "Eating chocolate every hour makes you immortal, study you can't see claims", "label": 1,
     "text": "A post references a study that supposedly proves eating chocolate every hour makes a person immortal, though the study cannot be located anywhere."},
    {"id": "f09", "title": "Secret cabal replaces the sun with a giant lightbulb, leak claims", "label": 1,
     "text": "A conspiracy post claims a secret cabal has replaced the sun with an enormous lightbulb and that ordinary people simply have not noticed the swap."},
    {"id": "f10", "title": "Phone batteries will explode if you blink twice, expert warns", "label": 1,
     "text": "A fake warning claims that smartphone batteries will instantly explode if the owner blinks twice while charging, attributed to an unnamed expert."},
    {"id": "f11", "title": "MIRACLE: Standing on one leg cures all diseases, ancient secret revealed", "label": 1,
     "text": "An article claims that standing on one leg for ten minutes cures every disease known to humanity, calling it an ancient secret suppressed by hospitals."},
    {"id": "f12", "title": "Politicians are actually lizards wearing human suits, viral video alleges", "label": 1,
     "text": "A viral video alleges, without evidence, that all politicians are actually lizard people wearing realistic human suits controlled from underground bunkers."},
    {"id": "f13", "title": "Microwaving your money doubles it instantly, finance hack goes viral", "label": 1,
     "text": "A viral finance hack claims that microwaving paper money for thirty seconds instantly doubles the amount, a physically impossible and false claim."},
    {"id": "f14", "title": "Breaking: gravity to be switched off next Tuesday, agency confirms", "label": 1,
     "text": "A hoax claims that a space agency confirmed gravity will be temporarily switched off next Tuesday and advises everyone to tie themselves down, which is impossible."},
    {"id": "f15", "title": "New app reads your dreams and sells them to advertisers tonight", "label": 1,
     "text": "An unfounded post claims a new app can record people's dreams while they sleep and sell them to advertisers the same night, with no technical basis."},
    {"id": "f16", "title": "Doctors hate this: one weird fruit regrows lost limbs in minutes", "label": 1,
     "text": "A clickbait ad claims that eating one weird fruit can regrow amputated limbs within minutes and that doctors are hiding this remedy from patients."},
    {"id": "f17", "title": "The ocean is secretly made of soda, divers reveal", "label": 1,
     "text": "A fabricated story claims that divers discovered the ocean is secretly made entirely of soda and that beverage companies have covered it up for decades."},
    {"id": "f18", "title": "Reading this headline gives you superpowers, neuroscientists astonished", "label": 1,
     "text": "A hoax claims that simply reading a particular headline permanently grants the reader superpowers, supposedly astonishing neuroscientists who cannot explain it."},
    {"id": "f19", "title": "City to replace all roads with trampolines next month, memo claims", "label": 1,
     "text": "A fake internal memo claims a city will replace every road with trampolines next month to reduce traffic, an announcement no official body has made."},
    {"id": "f20", "title": "Scientists confirm clouds are made of cotton candy you can harvest", "label": 1,
     "text": "A viral article claims scientists confirmed that clouds are made of cotton candy and that anyone can harvest them with a long pole, which is false."},
]

# Evidence snippets the fact-check retrieves over (id, text, source).
SEED_EVIDENCE: List[Dict] = [
    {"id": "e01", "source": "WHO", "text": "Drinking or injecting bleach or disinfectant is dangerous and can cause serious harm or death; it does not cure viral infections."},
    {"id": "e02", "source": "NASA", "text": "The Moon is a natural rocky satellite of Earth; it is not a hologram and has been visited by crewed and robotic missions."},
    {"id": "e03", "source": "Health authority", "text": "No pill can safely cause the loss of thirty pounds of body fat in a single day; rapid extreme weight loss claims are fraudulent."},
    {"id": "e04", "source": "Telecom regulator", "text": "5G is a radio communication technology for data transmission; it cannot control the weather or create storms."},
    {"id": "e05", "source": "Public health agency", "text": "Vaccines do not contain microchips; they contain antigens that train the immune system, and their ingredients are publicly documented."},
    {"id": "e06", "source": "Water utility", "text": "Treated tap water is tested for safety and does not transform people; claims of water causing zombification are baseless."},
    {"id": "e07", "source": "Astronomy reference", "text": "The Sun is a star powered by nuclear fusion; it is not a lightbulb and cannot be replaced by one."},
    {"id": "e08", "source": "Physics reference", "text": "Gravity is a fundamental force that cannot be switched off; objects with mass always attract one another."},
    {"id": "e09", "source": "Nutrition reference", "text": "Eating chocolate does not make a person immortal; no food confers immortality, and human lifespan has biological limits."},
    {"id": "e10", "source": "Medical reference", "text": "Humans cannot regrow amputated limbs by eating fruit; limb regeneration is not possible in humans with any known food."},
    {"id": "e11", "source": "Meteorology reference", "text": "Clouds are made of tiny water droplets and ice crystals, not cotton candy or any edible substance."},
    {"id": "e12", "source": "Oceanography reference", "text": "Ocean water is saline water composed of water and dissolved salts; it is not made of soda."},
    {"id": "e13", "source": "Cardiology journal", "text": "Regular moderate physical activity is associated with reduced cardiovascular disease risk in large cohort studies."},
    {"id": "e14", "source": "Immunization data", "text": "Childhood vaccination campaigns have reduced measles incidence; high vaccine coverage prevents outbreaks."},
    {"id": "e15", "source": "Hygiene guidance", "text": "Handwashing with soap reduces the transmission of many respiratory and gastrointestinal infections."},
    {"id": "e16", "source": "Finance reference", "text": "Heating paper currency does not increase its quantity or value; money cannot be physically duplicated by microwaving."},
]

# Claims with a gold verdict for fact-check evaluation (real = supported/true, fake = refuted/false).
SEED_CLAIMS: List[Dict] = [
    {"claim": "Drinking bleach cures every virus overnight.", "verdict": "fake"},
    {"claim": "5G towers secretly control the weather.", "verdict": "fake"},
    {"claim": "Vaccines contain mind-control microchips.", "verdict": "fake"},
    {"claim": "Gravity will be switched off next Tuesday.", "verdict": "fake"},
    {"claim": "Regular physical activity is associated with lower heart-disease risk.", "verdict": "real"},
    {"claim": "Handwashing with soap reduces the spread of infections.", "verdict": "real"},
    {"claim": "Childhood vaccination campaigns reduce measles cases.", "verdict": "real"},
    {"claim": "Clouds are made of cotton candy you can harvest.", "verdict": "fake"},
]


def news() -> List[Dict]:
    return [dict(x) for x in SEED_NEWS]


def evidence() -> List[Dict]:
    return [dict(x) for x in SEED_EVIDENCE]


def claims() -> List[Dict]:
    return [dict(x) for x in SEED_CLAIMS]


__all__ = ["SEED_NEWS", "SEED_EVIDENCE", "SEED_CLAIMS", "news", "evidence", "claims"]
