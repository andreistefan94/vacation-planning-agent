# Agent graph

Generated with `python -m vacation_agent.graph.orchestrator`. Dashed arrows = conditional routing (`Command(goto=...)`); `present_* -> choose_*/approve_*` pairs are the human-in-the-loop interrupts.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	start_turn(start_turn)
	check_requirements(check_requirements)
	resolve_places(resolve_places)
	present_destinations(present_destinations)
	choose_destination(choose_destination)
	delegate(delegate)
	build_variants(build_variants)
	present_change(present_change)
	approve_change(approve_change)
	present_overrun(present_overrun)
	approve_overrun(approve_overrun)
	present_variants(present_variants)
	choose_variant(choose_variant)
	generate_itinerary(generate_itinerary)
	present_final(present_final)
	final_approval(final_approval)
	finalize(finalize)
	respond(respond)
	__end__([<p>__end__</p>]):::last
	__start__ --> start_turn;
	approve_change -.-> build_variants;
	approve_change -.-> check_requirements;
	approve_change -.-> present_change;
	approve_change -.-> respond;
	approve_overrun -.-> check_requirements;
	approve_overrun -.-> present_overrun;
	approve_overrun -.-> present_variants;
	approve_overrun -.-> respond;
	build_variants -.-> present_change;
	build_variants -.-> present_overrun;
	build_variants -.-> present_variants;
	build_variants -.-> respond;
	check_requirements -.-> resolve_places;
	check_requirements -.-> respond;
	choose_destination -.-> check_requirements;
	choose_destination -.-> present_destinations;
	choose_destination -.-> respond;
	choose_variant -.-> build_variants;
	choose_variant -.-> check_requirements;
	choose_variant -.-> generate_itinerary;
	choose_variant -.-> present_variants;
	delegate -.-> build_variants;
	delegate -.-> respond;
	final_approval -.-> check_requirements;
	final_approval -.-> finalize;
	final_approval -.-> present_final;
	final_approval -.-> present_variants;
	finalize -.-> respond;
	generate_itinerary --> present_final;
	present_change --> approve_change;
	present_destinations --> choose_destination;
	present_final --> final_approval;
	present_overrun --> approve_overrun;
	present_variants --> choose_variant;
	resolve_places -.-> delegate;
	resolve_places -.-> present_destinations;
	resolve_places -.-> respond;
	start_turn -.-> check_requirements;
	start_turn -.-> respond;
	respond --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```
