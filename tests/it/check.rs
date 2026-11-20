//! The property check shared by all test tasks.

use planning_heuristics::{Combine, HAdd, HFf, HMax, Heuristic};
use sas_plus::Task;

use crate::oracle;

/// Checks every reachable state of `task`: values match the naive oracle, dead ends agree
/// across heuristics and with relaxed reachability, `h_max <= h*`, `h_max <= h_FF <= h_add`,
/// and the relaxed plan is valid with cost `h_FF`.
///
/// Returns the number of states checked.
pub fn check_task(name: &str, task: &Task) -> usize {
    task.validate().expect("test tasks are valid");
    let mut h_max = HMax::new(task);
    let mut h_add = HAdd::new(task);
    let mut h_ff = HFf::new(task);
    let perfect = oracle::perfect(task);

    for (state, h_star) in &perfect {
        let max = h_max.value(state);
        let add = h_add.value(state);
        let ff = h_ff.value(state);
        let ctx = || format!("{name}, state {:?}", state.0);

        assert_eq!(
            max,
            oracle::relaxed_goal_cost(task, state, Combine::Max),
            "h_max, {}",
            ctx()
        );
        assert_eq!(
            add,
            oracle::relaxed_goal_cost(task, state, Combine::Add),
            "h_add, {}",
            ctx()
        );
        assert_eq!(
            add,
            h_add.relaxation().explore(state, Combine::Add).goal_cost(),
            "early stop changes h_add, {}",
            ctx()
        );
        assert_eq!(max.is_none(), add.is_none(), "dead ends, {}", ctx());
        assert_eq!(max.is_none(), ff.is_none(), "dead ends, {}", ctx());
        assert_eq!(h_ff.estimate(state), ff.map(|v| u32::try_from(v).unwrap()));

        let (Some(max), Some(add), Some(ff)) = (max, add, ff) else {
            assert_eq!(
                *h_star,
                None,
                "relaxed dead end must be a real one, {}",
                ctx()
            );
            continue;
        };
        if let Some(h_star) = h_star {
            assert!(max <= *h_star, "h_max {max} > h* {h_star}, {}", ctx());
        }

        assert!(
            max <= ff && ff <= add,
            "h_max {max}, h_FF {ff}, h_add {add}, {}",
            ctx()
        );
        let plan = h_ff.relaxed_plan(state).expect("not a dead end");
        assert_eq!(
            Some(&plan),
            h_ff.relaxation().relaxed_plan(state).as_ref(),
            "reused and fresh workspaces disagree, {}",
            ctx()
        );
        assert!(
            oracle::is_relaxed_plan(task, state, plan.ops()),
            "invalid relaxed plan, {}",
            ctx()
        );
        assert_eq!(
            oracle::plan_cost(task, plan.ops()),
            ff,
            "plan cost, {}",
            ctx()
        );
    }

    perfect.len()
}
