"""
generate_dataset.py
Generates 500 Instagram caption training samples across 5 segments.
Output format: <SEG> Segment <DESC> Description -> Caption <END>
Saves to data/dataset.txt
"""

import random
import os

# ── Segment definitions ────────────────────────────────────────────────────────

SEGMENTS = {
    "Fitness": {
        "descs": [
            "Promote gym membership", "Morning workout routine", "Protein shake recipe",
            "Running motivation", "Yoga session", "Weight loss journey",
            "Strength training tips", "Home workout guide", "Marathon prep",
            "CrossFit challenge", "Cycling class", "HIIT session results",
        ],
        "verbs": ["crush", "power", "transform", "push", "build", "fuel",
                  "ignite", "unlock", "dominate", "elevate", "strengthen", "challenge"],
        "adjectives": ["stronger", "fitter", "unstoppable", "relentless", "powerful",
                       "lean", "energized", "fierce", "dedicated", "athletic"],
        "emojis": ["💪", "🏋️", "🔥", "🏃", "⚡", "🧘", "🥗", "🎯", "🏆", "💥"],
        "ctas": [
            "Join today and transform your body!",
            "Start your journey now!",
            "Drop a 💪 if you're in!",
            "Tag your gym buddy!",
            "Sign up — link in bio!",
            "Comment your PR below!",
            "Follow for daily workouts!",
            "Share with your squad!",
        ],
        "templates": [
            "{verb} your limits and get {adj}.",
            "Every rep counts. Get {adj} today.",
            "{verb} through the pain — results incoming.",
            "No excuses. Just results. Get {adj}.",
            "Your body is capable of anything. {verb} it.",
            "Rise before the sun. Train harder. Get {adj}.",
            "{verb} your potential — one workout at a time.",
        ],
    },
    "Food": {
        "descs": [
            "New brunch menu launch", "Homemade pasta recipe", "Vegan dessert idea",
            "Summer smoothie bowl", "Restaurant grand opening", "Street food tour",
            "Baking sourdough bread", "Healthy salad bowl", "Chocolate lava cake",
            "Sushi platter showcase", "Farm to table experience", "Spicy ramen bowl",
        ],
        "verbs": ["savor", "indulge", "taste", "devour", "relish", "explore",
                  "discover", "create", "enjoy", "celebrate", "bite", "feast"],
        "adjectives": ["delicious", "irresistible", "mouthwatering", "fresh", "golden",
                       "crispy", "creamy", "vibrant", "wholesome", "divine"],
        "emojis": ["🍕", "🥗", "🍜", "🧁", "🍣", "🥑", "🍓", "☕", "🍝", "🌮"],
        "ctas": [
            "Reserve your table — link in bio!",
            "Recipe dropping this week!",
            "Tag a foodie friend!",
            "Save this for later!",
            "Order now — link in bio!",
            "DM us for the recipe!",
            "Comment 🍕 to get the link!",
            "Follow for daily food inspo!",
        ],
        "templates": [
            "{verb} every bite of this {adj} creation.",
            "Life is too short for boring food. {verb} this.",
            "{adj} flavors that stop you mid-scroll.",
            "Made with love and served {adj}.",
            "{verb} the moment — one plate at a time.",
            "Food that makes you say 'wow'. Try this {adj} dish.",
            "{verb} your taste buds with something truly {adj}.",
        ],
    },
    "Fashion": {
        "descs": [
            "New summer collection drop", "Vintage thrift haul", "Sustainable fashion line",
            "Designer collab launch", "Street style lookbook", "Seasonal wardrobe refresh",
            "Capsule wardrobe guide", "Accessory collection reveal", "Denim edit launch",
            "Evening wear showcase", "Athleisure drop", "Monochrome outfit inspo",
        ],
        "verbs": ["wear", "style", "rock", "own", "define", "express",
                  "flaunt", "curate", "elevate", "showcase", "embrace", "redefine"],
        "adjectives": ["bold", "effortless", "timeless", "chic", "sleek",
                       "stunning", "elegant", "fierce", "iconic", "fresh"],
        "emojis": ["👗", "✨", "👠", "🕶️", "💅", "🎀", "👜", "🌟", "💎", "🖤"],
        "ctas": [
            "Shop the look — link in bio!",
            "Available now. Don't miss out!",
            "Tag someone who needs this!",
            "Drop a 🖤 if you love it!",
            "DM us for sizing info!",
            "Limited stock — grab yours!",
            "Follow for daily style drops!",
            "Save this outfit inspo!",
        ],
        "templates": [
            "{verb} your style. Make it {adj}.",
            "Fashion is self-expression. Make yours {adj}.",
            "{verb} the look that turns heads.",
            "Dress like every day is a {adj} occasion.",
            "{verb} what makes you feel {adj}.",
            "Style is attitude. Keep it {adj}.",
            "{verb} the room in something truly {adj}.",
        ],
    },
    "Travel": {
        "descs": [
            "Bali sunset adventure", "Paris weekend getaway", "Backpacking Southeast Asia",
            "New York city guide", "Santorini honeymoon", "Safari in Kenya",
            "Road trip through Iceland", "Tokyo street food tour", "Maldives resort stay",
            "Hiking the Himalayas", "Caribbean cruise review", "Amsterdam canal tour",
        ],
        "verbs": ["explore", "wander", "discover", "chase", "embrace", "journey",
                  "escape", "venture", "roam", "uncover", "experience", "seek"],
        "adjectives": ["breathtaking", "magical", "unforgettable", "stunning", "hidden",
                       "serene", "vibrant", "wild", "dreamy", "epic"],
        "emojis": ["✈️", "🌍", "🏖️", "🗺️", "🌅", "🏔️", "🌴", "🧳", "🌊", "🏛️"],
        "ctas": [
            "Plan your trip — link in bio!",
            "Save this for your bucket list!",
            "Tag your travel partner!",
            "Book now before it sells out!",
            "DM us for travel tips!",
            "Follow for weekly destination inspo!",
            "Drop a 🌍 if you've been here!",
            "Comment your dream destination!",
        ],
        "templates": [
            "{verb} every {adj} corner of the world.",
            "Not all who wander are lost. {verb} this {adj} place.",
            "{verb} the world — one {adj} destination at a time.",
            "Life is short. {verb} somewhere {adj}.",
            "{adj} views that stay with you forever.",
            "{verb} beyond the tourist trail. Find something {adj}.",
            "Collect moments, not things. This place is {adj}.",
        ],
    },
    "RealEstate": {
        "descs": [
            "Luxury penthouse listing", "Cozy studio apartment", "Beachfront villa for sale",
            "Downtown loft reveal", "Suburban family home", "Modern farmhouse listing",
            "Investment property tips", "Open house announcement", "New development launch",
            "Historic townhouse sale", "Mountain cabin listing", "Smart home showcase",
        ],
        "verbs": ["find", "discover", "unlock", "own", "invest", "secure",
                  "explore", "claim", "transform", "build", "imagine", "step into"],
        "adjectives": ["dream", "stunning", "luxurious", "modern", "spacious",
                       "breathtaking", "elegant", "cozy", "prime", "exclusive"],
        "emojis": ["🏡", "🔑", "🏙️", "✨", "🏠", "💰", "🌇", "🛋️", "🪴", "🏗️"],
        "ctas": [
            "Book a viewing — link in bio!",
            "DM us for more details!",
            "Schedule a tour today!",
            "Call us — contact in bio!",
            "Limited units available — act fast!",
            "Follow for new listings daily!",
            "Save this listing!",
            "Comment 🔑 to learn more!",
        ],
        "templates": [
            "{verb} your {adj} home today.",
            "Every family deserves a {adj} space. {verb} yours.",
            "{verb} into a life that feels {adj}.",
            "Home is where the heart is. Make it {adj}.",
            "{verb} the {adj} property you've always wanted.",
            "Your {adj} investment starts here.",
            "{verb} a space that inspires. Make it {adj}.",
        ],
    },
}

# ── Generator ──────────────────────────────────────────────────────────────────

def generate_sample(segment: str, config: dict) -> str:
    desc = random.choice(config["descs"])
    verb = random.choice(config["verbs"]).capitalize()
    adj  = random.choice(config["adjectives"])
    emoji = random.choice(config["emojis"])
    cta  = random.choice(config["ctas"])
    tmpl = random.choice(config["templates"])

    body = tmpl.format(verb=verb, adj=adj)
    caption = f"{body} {emoji} {cta}"

    return f"<SEG> {segment} <DESC> {desc} -> {caption} <END>"


def generate_dataset(n: int = 500, out_path: str = "data/dataset.txt") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    seg_names = list(SEGMENTS.keys())
    samples = []

    # Evenly distribute across segments
    per_seg = n // len(seg_names)
    remainder = n % len(seg_names)

    for i, seg in enumerate(seg_names):
        count = per_seg + (1 if i < remainder else 0)
        for _ in range(count):
            samples.append(generate_sample(seg, SEGMENTS[seg]))

    random.shuffle(samples)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(samples) + "\n")

    print(f"✅ Generated {len(samples)} samples → {out_path}")


if __name__ == "__main__":
    generate_dataset()
