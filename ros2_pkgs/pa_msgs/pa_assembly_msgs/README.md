# `pa_assembly_msgs`

The ExecuteMate action, and a separate package from `pa_gripper_msgs` on purpose.

`pa_gripper_msgs` carries the **parallel_jaw class contract** — what any gripper of that class
must serve. `ExecuteMate` is not a gripper action: it closes a mate declared by an ASSEMBLY, its
goal is an assembly-scoped address, and its refusals are about assembly graphs and occurrence
bindings. Putting it in the gripper package would give that package a name that misdescribes half
its contents, and the next reader would have to open it to find that out.

The end effector still appears in the surface — `ee_class`, and the mechanism table is keyed on
it — because which mates can be closed depends on what is attached. That is a reference to the
gripper contract, not a membership in it.

**Owed elsewhere:** a new interface package has to be staged where the others are. The authoring
image builds `pa_gripper_msgs` by name (`harness-integration/images/authoring-environment/`
Dockerfile + `stage-sources.sh`, and `harness_integration/constants.py`), and this package needs
the same treatment before an agent container can import the action. That is harness-integration's
lane and is flagged, not done here.
