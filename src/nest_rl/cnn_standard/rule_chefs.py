"""Rule-based partners for the native Overcooked soup task."""

from collections import Counter, deque

from overcooked_ai_py.agents.agent import Agent, AgentGroup
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import Recipe


ROLES = ("full", "prep", "serve")
INGREDIENT_TILES = {"onion": "O", "tomato": "T"}

# Some narrow layouts need one chef to let the other lead. The tuple order
# follows the player numbers in the native layout files.
LAYOUT_ROLES = {
    "bonus_order_test": ("prep", "serve"),
    "corridor": ("full", "serve"),
    "cramped_room_tomato": ("prep", "serve"),
    "m_shaped_s": ("full", "serve"),
    "multiplayer_schelling": ("prep", "prep", "prep", "serve"),
    "schelling": ("full", "serve"),
    "schelling_s": ("full", "serve"),
    "tutorial_1": ("serve", "full"),
    "tutorial_3": ("serve", "full"),
    "you_shall_not_pass": ("full", "serve"),
}


def make_rule_team(mdp, roles=None):
    """Create a native AgentGroup with the layout's verified role assignment."""
    if roles is None:
        roles = LAYOUT_ROLES.get(mdp.layout_name, ("full",) * mdp.num_players)
    if len(roles) != mdp.num_players:
        raise ValueError(f"{mdp.layout_name} requires {mdp.num_players} chef roles")
    team = AgentGroup(*(RuleChef(role) for role in roles))
    team.set_mdp(mdp)
    return team


class RuleChef(Agent):
    """Pick native actions from soup-production rules and the current layout."""

    def __init__(self, role="full"):
        if role not in ROLES:
            raise ValueError(f"Unknown chef role: {role}")
        super().__init__()
        self.role = role

    def set_mdp(self, mdp):
        super().set_mdp(mdp)
        self.floor = set(mdp.get_valid_player_positions())
        self.fixtures = set().union(*(mdp.terrain_pos_dict[tile] for tile in "XOTDPS"))
        self.counters = set(mdp.terrain_pos_dict["X"])
        self.pots = tuple(sorted(mdp.get_pot_locations()))
        self.serving = set(mdp.get_serving_locations())
        self.dishes = set(mdp.get_dish_dispenser_locations())
        self.sources = {
            name: set(mdp.terrain_pos_dict[tile]) for name, tile in INGREDIENT_TILES.items()
        }
        self.access = {}
        for position in sorted(self.floor):
            if position not in self.access:
                locations, targets = self._routes(position, Direction.NORTH, set())
                self.access.update(dict.fromkeys(locations, frozenset(targets)))

    def _routes(self, start, orientation, occupied):
        """Find the first step toward each interactable tile without crossing a chef."""
        pending = deque([start])
        locations = {start: (0, Action.STAY)}
        targets = {}
        while pending:
            position = pending.popleft()
            steps, first = locations[position]
            for direction in Direction.ALL_DIRECTIONS:
                neighbor = Action.move_in_direction(position, direction)
                if neighbor in self.floor and neighbor not in occupied and neighbor not in locations:
                    locations[neighbor] = steps + 1, first if steps else direction
                    pending.append(neighbor)
                if neighbor in self.fixtures:
                    turn = Action.INTERACT if orientation == direction else direction
                    targets.setdefault(neighbor, (steps + 1, first if steps else turn))
        return locations, targets

    def _next_order(self, state, contents):
        orders = sorted(
            state.all_orders,
            key=lambda order: (order not in state.bonus_orders, len(order.ingredients), order.ingredients),
        )
        for order in orders:
            required = Counter(order.ingredients)
            if self.mdp.old_dynamics and len(order.ingredients) != Recipe.MAX_NUM_INGREDIENTS:
                continue
            if required >= contents and all(self.sources[name] or contents[name] for name in required):
                return required
        return None

    def action(self, state):
        chef = state.players[self.agent_index]
        others = [player for index, player in enumerate(state.players) if index != self.agent_index]
        other_access = set().union(*(self.access[player.position] for player in others))
        access = self.access[chef.position]
        occupied = {player.position for player in others}
        _, routes = self._routes(chef.position, chef.orientation, occupied)
        handoff = self.counters & access & other_access
        free_handoff = handoff - state.objects.keys()
        held = chef.held_object.name if chef.has_object() else None
        peer_holds = Counter(player.held_object.name for player in others if player.has_object())

        def perform(*tasks, yield_for=()):
            for name, candidates in tasks:
                reachable = candidates & routes.keys()
                if reachable:
                    target = min(reachable, key=lambda tile: (routes[tile][0], tile))
                    return routes[target][1], {"task": name, "target": target}
            if yield_for:
                action = self._yield_action(chef, others, yield_for)
                if action != Action.STAY:
                    return action, {"task": "yield"}
            return Action.STAY, {"task": "wait"}

        def on_counter(kind):
            return {position for position, obj in state.objects.items()
                    if position in self.counters and obj.name == kind}

        # Prefer an active pot that can still produce a valid order. Some
        # tutorial starts have a pot preloaded with an incompatible recipe.
        for pot in sorted(self.pots, key=lambda tile: tile not in state.objects):
            soup = state.objects.get(pot)
            contents = Counter(soup.ingredients) if soup else Counter()
            order = self._next_order(state, contents)
            if order is not None and (soup is None or soup.is_idle or contents == order):
                break
        else:
            return perform()

        missing = order - contents
        pot_here = pot in access
        delivery_here = bool(access & self.serving)
        split = bool(others) and all(self.access[player.position] != access for player in others)
        cook_or_serve = self.role != "prep" or (split and pot_here and not other_access & self.dishes)
        staged_soup = on_counter("soup")

        if held == "soup":
            return perform(("deliver", self.serving), ("pass_soup", free_handoff if not delivery_here else set()))
        if held == "dish":
            if not pot_here:
                return perform(("pass_dish", free_handoff))
            return perform(("plate", {pot} if soup and soup.is_ready else set()))
        if held in self.sources:
            if pot_here and missing[held] and (soup is None or soup.is_idle):
                return perform(("add_ingredient", {pot}))
            return perform(("pass_ingredient" if not pot_here else "put_down", free_handoff))

        if staged_soup and delivery_here and (cook_or_serve or not other_access & self.serving):
            return perform(("take_soup", staged_soup))
        if staged_soup or peer_holds["soup"]:
            return perform()

        if soup and (soup.is_ready or soup.is_cooking):
            if pot_here and cook_or_serve and not peer_holds["dish"]:
                return perform(("take_dish", on_counter("dish") | self.dishes))
            if not pot_here and not on_counter("dish") and not peer_holds["dish"]:
                return perform(("take_dish_for_partner", self.dishes if free_handoff else set()))
            return perform(yield_for={pot} | self.dishes | self.serving)

        if soup and not missing and cook_or_serve and not self.mdp.old_dynamics:
            return perform(("start_cooking", {pot}))
        if not missing:
            return perform(yield_for={pot})

        prep_tasks = []
        for kind, count in missing.items():
            if self.role == "serve" and pot in other_access and self.sources[kind] & other_access:
                continue
            staged = on_counter(kind)
            if pot_here:
                prep_tasks.append(("take_staged_ingredient", staged))
            pending = peer_holds[kind] + len(staged)
            if count > pending and (pot_here or free_handoff):
                prep_tasks.append(("take_ingredient", self.sources[kind]))
        yield_for = {pot}.union(*(self.sources[kind] for kind in missing))
        return perform(*prep_tasks, yield_for=yield_for)

    def _yield_action(self, chef, others, targets):
        occupied = {player.position for player in others}

        def travel_cost(blocker):
            cost = 0
            for player in others:
                relevant = targets & self.access[player.position]
                if relevant:
                    blockers = (occupied - {player.position}) | {blocker}
                    _, routes = self._routes(player.position, player.orientation, blockers)
                    cost += sum(routes[tile][0] if tile in routes else len(self.floor) + 1
                                for tile in relevant)
            return cost

        best_cost = travel_cost(chef.position)
        best_action = Action.STAY
        for direction in Direction.ALL_DIRECTIONS:
            position = Action.move_in_direction(chef.position, direction)
            if position in self.floor and position not in occupied:
                cost = travel_cost(position)
                if cost < best_cost:
                    best_cost, best_action = cost, direction
        return best_action
