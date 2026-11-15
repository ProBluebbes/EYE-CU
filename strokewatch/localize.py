"""Where in the brain the signs MIGHT point: hedged screening wording, never a diagnosis.

Inputs are the signs StrokeWatch measures, each given as the patient's WEAK side (LEFT/RIGHT =
the patient's side, as everywhere else):
  face  - new lower-face droop (detector)
  arms / legs - one-sided drift or failure to lift (BE-FAST check)
  eyes  - gaze palsy or deviation (BE-FAST check). befast already stores the eye sign as the
          weak side a HEMISPHERIC lesion would give: eyes can't look left, or stay turned to
          the right -> weak side LEFT.

Rules from the clinical literature (simplified on purpose; imaging decides):
  * Hemispheric (cortical) stroke: the eyes deviate TOWARD the lesion and can't look away
    from it, i.e. away from the weak side; face, arm and leg weakness are on the side opposite
    the lesion. So signs on the patient's LEFT -> possible RIGHT hemispheric stroke.
    (Tijssen 1991 Neurology "Conjugate eye deviation: side, site and size of the hemispheric
    lesion"; Trobe, "Acute cerebral horizontal gaze deviation", NOVEL, Univ. of Utah.)
  * Gaze deviation with one-sided weakness is the strongest bedside predictor of a large-vessel
    occlusion (NIHSS gaze item ~58 % sensitive / ~95 % specific; RACE, FAST-ED and GPS scores
    use it). Cortical signs (gaze, aphasia, neglect) predict LVO better than weakness alone.
  * "Wrong-way" eyes, deviated TOWARD the weak side: pontine stroke (eyes turn away from a
    pontine lesion), also thalamic haemorrhage, or a seizure (eyes turn away from the focus).
  * A gaze palsy alone (can't look to one side) can also come from the pons on THAT side.
  * Crossed signs (face weak on one side, limbs on the other) point to the brainstem.
  * One-sided face/arm/leg weakness with no cortical signs also fits a small deep (lacunar)
    stroke in the internal capsule or pons.
  * Aphasia (here only indirect: instructions not followed) with right-sided weakness points
    to the language-dominant hemisphere, the left in most people (RACE scores aphasia with
    right hemiparesis, agnosia/neglect with left).
"""
from __future__ import annotations

CAVEAT = "Screening signs only, not a diagnosis: brain imaging and a clinician decide."


def _other(side: str) -> str:
    return "RIGHT" if side == "LEFT" else "LEFT"


def localize(face: str | None = None, exam: dict | None = None) -> dict | None:
    """Hedged localization from the face droop side and/or a BE-FAST check result
    (befast.ExamRunner.evaluate output). None if no one-sided sign was seen.

    Returns {"pattern", "weak_side", "hemisphere", "short", "text", "caveat"}:
      pattern    hemispheric | crossed | wrong_way_eyes | mixed
      weak_side  patient's weak side when the signs agree on one, else None
      hemisphere side of the brain the signs may point to (hemispheric pattern only)
      short      a few words for a banner, e.g. "may be a right hemispheric stroke"
      text       one or two sentences for staff
    """
    comps = (exam or {}).get("components") or {}

    def sign(k):
        c = comps.get(k) or {}
        return c.get("side") if c.get("status") == "abnormal" and c.get("side") in ("LEFT", "RIGHT") else None

    face = face if face in ("LEFT", "RIGHT") else None
    arm, leg, eyes = sign("arms"), sign("legs"), sign("eyes")
    no_response = bool((exam or {}).get("no_response"))
    limbs = {s for s in (arm, leg) if s}
    lw = lambda s: s.lower()

    def out(pattern, text, short, weak=None, hemi=None):
        return {"pattern": pattern, "weak_side": weak, "hemisphere": hemi, "short": short,
                "text": text, "caveat": CAVEAT}

    if not (face or limbs or eyes):
        return None
    if len(limbs) == 2:
        return out("mixed", "Arm and leg weakness on different sides: the signs don't point to "
                            "one side of the brain. This may be a brainstem stroke, or have "
                            "another cause.", "signs on both sides")
    limb = next(iter(limbs), None)
    limb_word = " and ".join(w for w, s in (("arm", arm), ("leg", leg)) if s)
    if limb and face and face != limb:
        return out("crossed", f"Facial droop on the {lw(face)} but {limb_word} weakness on the "
                              f"{lw(limb)} (crossed signs): this may be a brainstem stroke.",
                   "may be a brainstem stroke")
    if eyes and limb and eyes != limb:
        return out("wrong_way_eyes",
                   f"The eyes turn toward the weak {lw(limb)} side (\"wrong-way\" eyes), the "
                   f"opposite of a typical hemispheric stroke: this may be a brainstem (pontine) "
                   f"stroke; a thalamic bleed or a seizure can look the same.",
                   "may be a brainstem stroke or a seizure")
    if eyes and face and eyes != face:  # no limb sign to break the tie
        return out("mixed", f"Facial droop on the {lw(face)} but the eye signs point to the "
                            f"{lw(eyes)} side: the signs don't point to one side of the brain. "
                            f"This may be a brainstem stroke, or have another cause.",
                   "signs on both sides")

    weak = limb or face or eyes
    hemi = _other(weak)
    seen = [w for w, s in (("facial droop", face), (f"{limb_word} weakness", limb),
                           ("eye signs", eyes)) if s]
    seen_txt = seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " and " + seen[-1]
    text = (f"{seen_txt[0].upper()}{seen_txt[1:]} on the patient's {lw(weak)} side: "
            f"this may be a {lw(hemi)} hemispheric stroke.")
    if eyes and (limb or face):
        text += (" Eye signs together with one-sided weakness are a warning sign of a large "
                 "stroke (large-vessel occlusion).")
    elif eyes:
        text += (f" Less often, a brainstem stroke on the {lw(weak)} side causes the same eye "
                 f"sign.")
    else:
        what = "A droop alone" if not limb else "One-sided weakness without eye signs"
        text += (f" {what} can also come from a small deep (lacunar) or brainstem stroke"
                 + (", or a facial nerve problem." if not limb else "."))
    if no_response and hemi == "LEFT":
        text += (" Not following instructions may reflect a language problem (aphasia), which "
                 "fits a left hemispheric stroke.")
    return out("hemispheric", text, f"may be a {lw(hemi)} hemispheric stroke", weak, hemi)


def _selftest():
    def exam(**sides):
        return {"components": {k: {"status": "abnormal", "side": s} for k, s in sides.items()}}

    # can't look LEFT (befast: eyes side LEFT) -> RIGHT hemisphere, not "left-sided stroke"
    r = localize(exam=exam(eyes="LEFT"))
    assert r["hemisphere"] == "RIGHT" and "right hemispheric" in r["short"], r
    r = localize("LEFT", exam(eyes="LEFT", arms="LEFT"))
    assert r["hemisphere"] == "RIGHT" and "large-vessel" in r["text"], r
    r = localize("RIGHT")
    assert r["hemisphere"] == "LEFT" and "lacunar" in r["text"], r
    r = localize(None, {**exam(arms="RIGHT"), "no_response": True})
    assert r["hemisphere"] == "LEFT" and "aphasia" in r["text"], r
    assert localize("LEFT", exam(arms="RIGHT"))["pattern"] == "crossed"
    assert localize(None, exam(eyes="RIGHT", arms="LEFT"))["pattern"] == "wrong_way_eyes"
    assert localize(None, exam(arms="LEFT", legs="RIGHT"))["pattern"] == "mixed"
    assert localize(None, {"components": {"eyes": {"status": "inconclusive", "side": None}}}) is None
    for r in (localize(exam=exam(eyes="LEFT")), localize("RIGHT")):
        print(r["short"], "|", r["text"])
    print("localize self-test OK")


if __name__ == "__main__":
    _selftest()
