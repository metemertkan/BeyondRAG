"""Episode 8 claims: Einstein's June 1905 paper in plain words, and two control groups.

E  Einstein's claims, one per idea, in paper order. `quote` is the passage it comes from, in the
   1920 English edition (Saha's translation, Wikisource). The claims are my wording of his ideas.
K  Already in print before 30 June 1905. A working checker should support these. `source` names
   where it was printed.
W  Wrong. `kind` is "invented" (an error I made up) or "of the day" (what physics believed in 1905).
   The second kind is the trap: a checker grounded in 1905 should support it.

Each claim has two phrasings in words. Where the paper gives a formula, a third phrasing ("f")
writes it as a formula, to show how the checker treats formulas.
"""

from __future__ import annotations

from typing import Dict, List

CLAIMS: List[Dict] = [
    # ---------------- E: Einstein, June 1905 ----------------
    dict(id="E01", group="E", section="Introduction",
         quote="The observed phenomena in this case depend only on the relative motion of the conductor and the magnet, while according to the usual conception, a distinction must be made between the cases where the one or the other of the bodies is in motion.",
         phrasings=["The usual theory explains the current set up by a moving magnet in a different way from the current set up by a moving conductor, although the effect depends only on their relative motion.",
                    "Whether the magnet moves past the conductor or the conductor moves past the magnet, only the relative motion matters, yet the accepted theory treats the two cases differently."]),
    dict(id="E02", group="E", section="Introduction",
         quote="no properties of observed facts correspond to a concept of absolute rest",
         phrasings=["No observed phenomenon corresponds to the idea of absolute rest.",
                    "No experiment, whether mechanical, optical or electrical, can show that a body is absolutely at rest."]),
    dict(id="E03", group="E", section="Introduction",
         quote="for all coordinate systems for which the mechanical equations hold, the equivalent electrodynamical and optical equations hold also",
         phrasings=["The same laws of electrodynamics and optics hold in every system of co-ordinates in which the laws of mechanics hold.",
                    "The equations of electricity and of light keep the same form for every observer in uniform motion, just as the equations of mechanics do."]),
    dict(id="E04", group="E", section="Introduction, § 2",
         quote="Every ray of light moves in the \"stationary co-ordinate system\" with the same velocity c ... light, when measured in the moving system is always propagated with the constant velocity c",
         phrasings=["The velocity of light in empty space is the same for every observer in uniform motion, whatever the motion of the observer or of the source.",
                    "An observer moving with any uniform velocity measures the same velocity of light in empty space as an observer at rest."]),
    dict(id="E05", group="E", section="Introduction",
         quote="The introduction of a \"Lightäther\" will be proved to be superfluous, for according to the conceptions which will be developed, we shall introduce neither a space absolutely at rest",
         phrasings=["The introduction of a luminiferous ether is superfluous.",
                    "The theory of light and electricity needs neither an ether nor a space absolutely at rest."]),
    dict(id="E06", group="E", section="§ 1",
         quote="This last time (i.e., common time) can be defined, if we establish by definition that the time which light requires in travelling from A to B is equivalent to the time which light requires in travelling from B to A.",
         phrasings=["Clocks at two distant places are set to the same time by defining that light takes as long to go from the first place to the second as to come back.",
                    "The time at a distant place has a meaning only after clocks there have been set by exchanging light signals, taking the time out and the time back to be equal."]),
    dict(id="E07", group="E", section="§ 2",
         quote="two events which are synchronous when viewed from one system, will not be synchronous when viewed from a system moving relatively to this system",
         phrasings=["Two events which are simultaneous for one observer are not simultaneous for an observer who moves relatively to the first.",
                    "Simultaneity has no absolute meaning: whether two distant events happen at the same time depends on the motion of the observer."]),
    dict(id="E08", group="E", section="§ 2, § 4",
         quote="This length is to be estimated on the basis of our principle, and we shall find it to be different from l. ... It is clear that similar results hold about stationary bodies in a stationary system when considered from a uniformly moving system.",
         phrasings=["The shortening of moving bodies is mutual: each of two observers in relative motion finds the measuring-rods of the other shortened.",
                    "A rod at rest appears shortened to an observer moving past it, just as a moving rod appears shortened to an observer at rest."]),
    dict(id="E09", group="E", section="§ 2-4",
         quote="Let us have a rigid rod at rest ... this can be obtained by either of these operations ... we shall find it to be different from l.",
         phrasings=["The shortening of a moving body follows from the way lengths and times are measured, and not from any force acting on its molecules.",
                    "The contraction of moving bodies is a consequence of the definition of simultaneity, not a physical change caused by motion through the ether."]),
    dict(id="E10", group="E", section="§ 3",
         quote="now the problem is to find out the system of equations connecting these magnitudes ... These two assumptions are quite sufficient to give us a simple and consistent theory",
         phrasings=["The equations connecting the co-ordinates and the time of two systems in uniform relative motion follow from the principle of relativity and the constancy of the velocity of light alone.",
                    "The transformation of co-ordinates and time between moving systems can be derived from two principles only, without any hypothesis about the constitution of matter or of the ether."]),
    dict(id="E11", group="E", section="§ 3",
         quote="Therefore the wave is propagated in the moving system with the same velocity c, and as a spherical wave. Therefore we show that the two principles are mutually reconcilable.",
         phrasings=["A spherical wave of light which spreads with the velocity of light in one system is also a spherical wave spreading with the same velocity in a system moving relatively to it.",
                    "The principle of relativity and the constancy of the velocity of light do not contradict each other: a light wave is spherical for both of two observers in relative motion."]),
    dict(id="E12", group="E", section="§ 4",
         quote="For v = c, all moving bodies, when considered from a stationary system shrink into planes. For a velocity larger than the velocity of light, our propositions become meaningless",
         phrasings=["At the velocity of light all moving bodies would shrink into planes, and velocities greater than that of light have no meaning.",
                    "No body can move faster than light; at the velocity of light a moving body would be flattened completely."]),
    dict(id="E13", group="E", section="§ 4",
         quote="Therefore the clock loses by an amount ½ v²/c² per second of motion, to the second order of approximation.",
         phrasings=["A clock in motion goes more slowly than an identical clock at rest.",
                    "Seen from a system at rest, a moving clock loses time, and loses more the faster it moves."],
         formula="A clock moving with velocity v loses ½ v²/c² of a second for every second, compared with a clock at rest."),
    dict(id="E14", group="E", section="§ 4",
         quote="if we set in motion one of them with a constant velocity along a closed curve till it comes back to A ... the last mentioned clock will be behind the stationary one",
         phrasings=["If one of two clocks which go together is carried round a closed path and brought back, it is found to be behind the clock which stayed at rest.",
                    "A clock that makes a journey and returns shows less time elapsed than an identical clock that never moved."]),
    dict(id="E15", group="E", section="§ 4",
         quote="a clock placed at the equator must be slower by a very small amount than a similarly constructed clock which is placed at the pole",
         phrasings=["A clock at the equator goes very slightly slower than an identical clock at the pole.",
                    "Because of the earth's rotation, a clock placed at the equator runs a little slow compared with the same kind of clock at the pole."]),
    dict(id="E16", group="E", section="§ 5",
         quote="by combining two velocities, each of which is smaller than c, we obtain a velocity which is always smaller than c",
         phrasings=["Two velocities, each smaller than the velocity of light, combine to give a velocity which is still smaller than the velocity of light.",
                    "Velocities do not simply add: adding two velocities below that of light never gives a velocity equal to or above that of light."],
         formula="A body moving with velocity w in a system which itself moves with velocity v has the velocity (v + w)/(1 + vw/c²), not v + w."),
    dict(id="E17", group="E", section="§ 5",
         quote="It is also clear that the velocity of light c cannot be altered by adding to it a velocity smaller than c.",
         phrasings=["The velocity of light cannot be increased by adding to it the velocity of a moving body.",
                    "Light sent out from a moving system still has the velocity of light, not that velocity plus the velocity of the system."]),
    dict(id="E18", group="E", section="§ 6",
         quote="the electric and magnetic forces possess no existence independent of the nature of motion of the co-ordinate system",
         phrasings=["Electric and magnetic forces have no existence independent of the state of motion of the observer.",
                    "A field which is purely electric for one observer is partly magnetic for another observer moving relatively to the first."]),
    dict(id="E19", group="E", section="§ 6",
         quote="the assymetry mentioned in the introduction which occurs when we treat of the current excited by the relative motion of a magnet and a conductor disappears",
         phrasings=["Once the forces are referred to a system moving with the conductor, the different explanations for a moving magnet and a moving conductor become one.",
                    "The current induced by a magnet and a conductor in relative motion needs only one explanation, whichever of the two moves."]),
    dict(id="E20", group="E", section="§ 8",
         quote="It is to be noticed that the energy and the frequency of a light-complex vary according to the same law with the state of motion of the observer.",
         phrasings=["The energy and the frequency of a ray of light change in the same ratio when the observer moves.",
                    "For an observer in motion, the energy of a portion of light changes in exactly the same proportion as its frequency."]),
    dict(id="E21", group="E", section="§ 9",
         quote="if an electrically charged body moves in any manner in space, and if its charge does not change thereby, when regarded from a system moving along with it, then the charge remains constant even when it is regarded from the stationary system",
         phrasings=["The electric charge of a body is the same for every observer, whatever the motion of the body.",
                    "A charge which does not change for an observer moving with the body does not change for an observer at rest either."]),
    dict(id="E22", group="E", section="§ 10",
         quote="For v=c, W is infinitely great. ... In consequence of the arguments mentioned above, this expression for kinetic energy must also hold for ponderable masses.",
         phrasings=["The kinetic energy of any body, charged or not, becomes infinitely great as its velocity approaches the velocity of light.",
                    "The rules for the mass and energy of a moving electron hold for all ponderable bodies, and no body can be brought to the velocity of light."],
         formula="The kinetic energy of a body of mass m moving with velocity v is mc²(1/√(1 − v²/c²) − 1), which becomes infinite when v = c."),
    # ---------------- K: in print before 30 June 1905 ----------------
    dict(id="K01", group="K", source="Michelson and Morley 1887",
         phrasings=["Optical experiments made to detect the motion of the earth through the ether have given a negative result.",
                    "The interference experiment of Michelson and Morley did not show the displacement expected from the earth's motion through the ether."]),
    dict(id="K02", group="K", source="Poincaré, St. Louis lecture, 1904",
         phrasings=["The laws of physical phenomena must be the same for an observer at rest and for an observer carried along in a uniform motion of translation.",
                    "We have no means of discovering whether or not we are carried along in a uniform motion of translation."]),
    dict(id="K03", group="K", source="FitzGerald 1889, Lorentz 1892-1904",
         phrasings=["A body moving through the ether is shortened in the direction of its motion by an amount depending on the square of the ratio of its velocity to the velocity of light.",
                    "To explain the result of Michelson and Morley, it was supposed that material bodies contract slightly in the direction in which they move through the ether."]),
    dict(id="K04", group="K", source="Lorentz 1895, 1899, 1904",
         phrasings=["For a system moving through the ether, a local time is introduced which differs from the general time by an amount depending on the position of the point.",
                    "Lorentz used a modified time, called local time, which is different at different points of a moving system."]),
    dict(id="K05", group="K", source="Poincaré 1900, 1904",
         phrasings=["Clocks at distant stations are set by exchanging light signals, and if the stations are moving, the clocks so set show local time.",
                    "When two observers in motion set their watches by signals of light, their watches mark the local time and not the true time."]),
    dict(id="K06", group="K", source="Poincaré, The Measure of Time, 1898",
         phrasings=["We have no direct intuition of the simultaneity of two events which happen in different places.",
                    "Saying that two distant events happen at the same moment requires a convention; we have no direct sense of it."]),
    dict(id="K07", group="K", source="Kaufmann 1901-1903, Abraham 1902",
         phrasings=["The mass of an electron increases with its velocity.",
                    "Experiments on the rays of radium show that the mass of the swiftest electrons is greater than that of slow ones."]),
    dict(id="K08", group="K", source="Poincaré, St. Louis lecture, 1904",
         phrasings=["In the new mechanics, no velocity could surpass the velocity of light.",
                    "If the new results on electrons are confirmed, the velocity of light would become a limit which no body could exceed."]),
    dict(id="K09", group="K", source="the wave theory of light",
         phrasings=["In the wave theory, the velocity of light does not depend on the velocity of the source which emits it.",
                    "Light from a moving source travels through the ether with the same velocity as light from a source at rest."]),
    dict(id="K10", group="K", source="Trouton and Noble 1903",
         phrasings=["A charged condenser moving with the earth shows no tendency to turn, though the theory of a stationary ether predicted one.",
                    "Trouton and Noble found no couple acting on a charged condenser carried along by the motion of the earth."]),
    dict(id="K11", group="K", source="Lorentz 1904",
         phrasings=["The equations of the electromagnetic field keep their form in a moving system if new variables are introduced for the co-ordinates, the time and the forces.",
                    "By a change of variables for space, time and the field, the electromagnetic equations for a moving system can be brought to the form they have for a system at rest."]),
    dict(id="K12", group="K", source="Poincaré, Science and Hypothesis, 1902",
         phrasings=["There is no absolute space, and there is no absolute time.",
                    "We only conceive of relative motion; absolute space and absolute time have no meaning for us."]),
    # ---------------- W: wrong ----------------
    dict(id="W01", group="W", kind="invented",
         phrasings=["The velocity of light in empty space is the velocity of the body which emits it added to the ordinary velocity of light.",
                    "Light from a source moving towards us reaches us faster than light from a source at rest, by the velocity of the source."]),
    dict(id="W02", group="W", kind="invented",
         phrasings=["A body moving through the ether becomes longer in the direction of its motion.",
                    "Motion through the ether stretches a rod in the direction in which it moves."]),
    dict(id="W03", group="W", kind="invented",
         phrasings=["The experiment of Michelson and Morley showed the full displacement of the interference fringes required by the earth's motion through the ether.",
                    "Optical experiments have measured the velocity of the earth through the ether."]),
    dict(id="W04", group="W", kind="invented",
         phrasings=["The mass of an electron decreases as its velocity increases.",
                    "The swiftest electrons from radium have a smaller mass than slow ones."]),
    dict(id="W05", group="W", kind="invented",
         phrasings=["If one of two clocks which go together is carried round a closed path and brought back, it is found to be ahead of the clock which stayed at rest.",
                    "A clock that makes a journey and returns shows more time elapsed than an identical clock that never moved."]),
    dict(id="W06", group="W", kind="invented",
         phrasings=["Two velocities, each smaller than the velocity of light, can combine to give a velocity several times greater than the velocity of light.",
                    "By adding the velocity of a moving system to the velocity of a body within it, any velocity whatever can be reached, even beyond that of light."]),
    dict(id="W07", group="W", kind="of the day",
         phrasings=["Light consists of waves in a luminiferous ether which fills all space.",
                    "The ether is the medium which carries light and electric and magnetic actions."]),
    dict(id="W08", group="W", kind="of the day",
         phrasings=["The ether remains at rest while the earth moves through it.",
                    "The ether is stationary, and material bodies move through it without carrying it along."]),
    dict(id="W09", group="W", kind="of the day",
         phrasings=["Local time is only an apparent time; behind it there is a true time, the same for all observers.",
                    "Clocks set by light signals in a moving system show a mere auxiliary time, and the true time is that of the ether."]),
    dict(id="W10", group="W", kind="of the day",
         phrasings=["The contraction of moving bodies is caused by the action of the ether on the molecular forces of the body.",
                    "Bodies contract when moving through the ether because the forces between their molecules are altered by the motion."]),
    dict(id="W11", group="W", kind="of the day",
         phrasings=["The mass of the electron is entirely of electromagnetic origin.",
                    "All the inertia of an electron comes from its electric field; it has no other mass."]),
    dict(id="W12", group="W", kind="of the day",
         phrasings=["The electron is a rigid sphere which keeps its spherical shape when it moves.",
                    "A moving electron does not contract, but remains a perfect sphere at every velocity."]),
]


def phrasings() -> List[Dict]:
    """One row per phrasing to check."""
    rows = []
    for claim in CLAIMS:
        for n, text in enumerate(claim["phrasings"], start=1):
            rows.append({"claim": claim["id"], "group": claim["group"], "phrasing": str(n), "text": text})
        if claim.get("formula"):
            rows.append({"claim": claim["id"], "group": claim["group"], "phrasing": "f", "text": claim["formula"]})
    return rows


if __name__ == "__main__":
    from collections import Counter

    rows = phrasings()
    print(Counter(c["group"] for c in CLAIMS), len(rows), "phrasings")
