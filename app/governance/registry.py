from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PropertySpec:
    key: str
    aliases: tuple[str, ...]
    cardinality: str = "single"
    mutable: bool = True
    default_scope: str = "general"
    category: str = "profile"


PROPERTIES = (
    PropertySpec("employer", ("employer", "company", "work at", "work for", "currently work", "雇主", "公司", "任职", "工作单位", "哪里工作"), default_scope="work", category="work"),
    PropertySpec("residence", ("residence", "live", "living", "address", "住址", "居住", "住在", "地址", "居所"), default_scope="home", category="life"),
    PropertySpec("job_role", ("job_role", "job title", "position", "职位", "岗位", "职务"), default_scope="work", category="work"),
    PropertySpec("project_status", ("project_status", "project status", "progress", "项目状态", "进展", "阶段"), category="project"),
    PropertySpec("deadline", ("deadline", "due date", "截止", "期限"), category="project"),
    PropertySpec("preference", ("preference", "prefer", "like", "偏好", "喜欢"), cardinality="multi", category="preference"),
    PropertySpec("skills", ("skills", "skill", "技能"), cardinality="multi"),
    PropertySpec("rules", ("rules", "rule", "规则", "例外", "exception"), cardinality="multi", mutable=False, category="rules"),
    PropertySpec("event", ("event", "事件", "决定", "decision"), cardinality="multi", mutable=False, category="events"),
)


def property_spec(key: str) -> PropertySpec:
    folded = key.strip().casefold()
    for spec in PROPERTIES:
        if folded == spec.key or folded in spec.aliases:
            return spec
    return PropertySpec(folded, (folded,), "unknown", False, category="other")
