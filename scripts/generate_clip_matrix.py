import torch
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

# 150 个 ADE20K 类别及其纯视觉描述
class_descriptions = {
    "wall": "Vertical, flat surface forming boundaries; typically solid colors with textures ranging from smooth plaster to rough masonry or brick patterns.",
    "building": "Large, upright multi-story structure with geometric lines, windows, and doors; exterior surfaces of stone, brick, glass, or concrete.",
    "sky": "Vast, continuous expanse at the top of an image; typically blue, white, or gray, often containing soft, fluffy cloud textures.",
    "floor": "Horizontal, flat surface at the bottom of indoor scenes; often made of wood planks, stone tiles, or smooth carpet material.",
    "tree": "Tall structure with a brown, rough-textured cylindrical trunk branching into a dense canopy of green, organic foliage or bare twigs.",
    "ceiling": "Horizontal, flat surface at the top of an indoor scene; typically solid white or off-white with a smooth or slightly grainy texture.",
    "road": "Wide, flat, linear path for vehicles; typically dark gray asphalt or concrete with a weathered, grainy surface and painted lane markings.",
    "bed": "Large, rectangular horizontal furniture covered in soft, fabric-textured layers like sheets and blankets, usually elevated on a frame with pillows at one end.",
    "windowpane": "Flat, rectangular sheet of transparent or reflective glass held within a frame, often showing a view of the outdoors or reflecting interior light.",
    "grass": "Densely packed, thin green blades forming an organic, uneven carpet-like texture on the ground in outdoor environments.",
    "cabinet": "Box-like storage unit with flat vertical doors; typically made of wood or metal with a smooth finish and small protruding handles.",
    "sidewalk": "Narrow, flat pedestrian path adjacent to a road; usually light gray concrete with a grid-like pattern of expansion joints.",
    "person": "Upright biological form with a distinct head, torso, and limbs; covered in varied fabric textures and colors from clothing.",
    "earth": "Dark brown or tan-colored ground surface with a rough, granular, and organic texture, often containing rocks or scattered dry debris.",
    "door": "Solid, rectangular vertical panel within a wall frame; usually made of wood or metal with a smooth finish and a circular handle.",
    "table": "Table with a flat, horizontal surface supported by four cylindrical or rectangular legs; typically made of wood, glass, or metal.",
    "mountain": "Massive, sloped landform rising to a peak; rugged, rocky texture with colors ranging from gray and brown to snow-capped white.",
    "plant": "Small to medium organic form with green leaves and thin stems, often growing in a pot or from the ground.",
    "curtain": "Soft, vertically hanging fabric panels flanking a window; often draped in folds with various colors, patterns, and translucent or opaque textures.",
    "chair": "Furniture with a flat horizontal seat, an upright backrest, and four legs; made of wood, metal, molded plastic, or upholstered fabric.",
    "car": "Metallic, reflective box-like vehicle with rubber tires, glass windows, and a streamlined aerodynamic shape; found on roads or in parking lots.",
    "water": "Transparent or reflective fluid surface; typically blue, green, or gray with rippled, wavy, or smooth textures depending on movement.",
    "painting": "Flat, rectangular frame containing colorful, patterned, or representational imagery; typically hung vertically on a wall surface.",
    "sofa": "Large, elongated upholstered furniture with soft cushions on the seat and back, usually with armrests at both ends.",
    "shelf": "Thin, flat horizontal board attached to a wall or within a cabinet for supporting objects; typically made of wood or metal.",
    "house": "Medium-sized standalone building with a sloped roof, windows, and an entrance; exterior usually made of wood, brick, or stucco.",
    "sea": "Deep blue or turquoise expanse of water stretching to the horizon, characterized by constant wave patterns and white foam at the edges.",
    "mirror": "Smooth, flat glass surface in a frame that provides a clear, high-contrast reflection of surrounding objects and light.",
    "rug": "Textured fabric or fiber covering for a floor; usually rectangular with various colors, decorative patterns, and a soft, pile-like surface.",
    "field": "Expansive, flat open land covered in a uniform layer of low-growing plants, crops, or grass, typically green or golden-brown.",
    "armchair": "Single-person upholstered chair with large, padded armrests and a soft, cushioned back and seat.",
    "seat": "Any flat, horizontal surface intended for sitting, often part of a larger structure like a bench or vehicle interior.",
    "fence": "Vertical barrier made of repeating upright wood slats, metal wires, or horizontal rails; often surrounds a yard or field.",
    "desk": "Flat, rectangular working surface, often with drawers underneath, supported by legs or side panels; usually made of wood or metal.",
    "rock": "Irregular, solid mass of mineral matter with a hard, rough, or jagged texture; typically gray, brown, or tan.",
    "wardrobe": "Tall, large box-like wooden cabinet with vertical doors for hanging and storing large items; often found in bedrooms.",
    "lamp": "Illumination device with a base, a vertical neck, and a fabric or glass shade, usually positioned on a table or floor.",
    "bathtub": "Large, elongated white ceramic or acrylic basin with a rounded interior, typically found fixed to a bathroom floor or wall.",
    "railing": "Horizontal bar supported by vertical posts; usually made of metal or wood, located along stairs, balconies, or bridges.",
    "cushion": "Soft, rectangular or square fabric bag filled with padding, used for comfort on chairs, sofas, or floors.",
    "base": "The lowermost supporting part of an object or column, typically wider than the rest of the structure.",
    "box": "Rigid, rectangular-prism-shaped container made of cardboard, wood, or plastic with flat sides and sharp corners.",
    "column": "Tall, vertical cylindrical pillar used as a support or decoration; often made of stone, concrete, or metal.",
    "signboard": "Flat, rectangular board containing text or graphics, usually mounted on a post, building, or hanging above a street.",
    "chest of drawers": "Large, rectangular wooden furniture housing several stacked horizontal sliding drawers with small handles.",
    "counter": "Long, flat horizontal surface in a kitchen or shop; usually made of smooth stone, laminate, or wood.",
    "sand": "Fine, granular yellow or tan-colored sediment found on beaches or deserts, forming smooth dunes or rippled patterns.",
    "sink": "Small, rounded or rectangular basin with a faucet, made of white ceramic or shiny stainless steel, embedded in a counter.",
    "skyscraper": "Extremely tall building with many stories, characterized by a vertical, rectangular profile and a grid of numerous glass windows.",
    "fireplace": "Rectangular opening at the base of a wall, usually framed by stone or brick, sometimes containing red and orange glowing embers.",
    "refrigerator": "Tall, box-like metallic appliance with one or two vertical doors, usually white, silver, or black with a smooth finish.",
    "grandstand": "Tiered seating structure made of repeating horizontal rows of benches or chairs, typically found in staduims.",
    "path": "Narrow, unpaved strip of ground for walking, often made of packed earth, gravel, or mown grass.",
    "stairs": "Sequence of repeating horizontal steps and vertical risers, creating a sloped path between different floor levels.",
    "runway": "Long, wide, flat strip of dark asphalt or concrete with distinct painted markings and lights along the edges.",
    "case": "Hard, boxy container with a handle, usually made of leather, plastic, or metal, used for carrying items.",
    "pool table": "Large, rectangular table covered in a flat green fabric, surrounded by raised wooden edges and supported by thick legs.",
    "pillow": "Rectangular or square bag of soft fabric filled with padding, typically resting at the head of a bed.",
    "screen door": "Lightweight door frame containing a secondary fine mesh of metal or plastic wire for ventilation.",
    "stairway": "Vertical passage containing a set of stairs, often enclosed by walls or defined by a railing.",
    "river": "Continuous, linear flow of water with organic, curved banks; typically blue-brown in color with rippled or flowing textures.",
    "bridge": "Elevated structure spanning water or a road, made of reinforced concrete, steel girders, or stone arches.",
    "bookcase": "Large furniture unit with multiple horizontal shelves for storing books, typically made of wood and placed against a wall.",
    "blind": "Window covering made of horizontal or vertical slats of wood, plastic, or metal that can be tilted or drawn up.",
    "coffee table": "Low, small rectangular table typically placed in front of a sofa; made of wood, glass, or metal.",
    "toilet": "White ceramic fixture with a rounded bowl and a vertical tank, typically fixed to a bathroom floor.",
    "flower": "Small, colorful organic structure with a center and delicate petals, growing from a green stem with leaves.",
    "book": "Stack of rectangular paper sheets bound together with a thicker, colorful cover; usually seen on shelves or tables.",
    "hill": "Rounded, elevated area of land, usually covered in green grass or brown soil, smaller and smoother than a mountain.",
    "bench": "Long seat for multiple people, often made of slats of wood or metal, commonly found in parks or public spaces.",
    "countertop": "Smooth, horizontal work surface finishing a kitchen cabinet; often made of granite, marble, or laminate.",
    "stove": "Box-like kitchen appliance with flat circular burners on top and a front-opening door for an oven.",
    "palm": "Type of tree with a tall, slender, unbranched trunk and a crown of large, fan-like or feathery green fronds.",
    "kitchen island": "Freestanding cabinet unit in the center of a kitchen with a flat countertop and storage below.",
    "computer": "Electronic device consisting of a flat rectangular monitor screen and a smaller box-like case, often accompanied by a keyboard.",
    "swivel chair": "Office chair with a backrest and armrests, mounted on a central post with a multi-legged wheeled base.",
    "boat": "Vessel for traveling on water, characterized by a curved, hollow hull and a pointed bow.",
    "bar": "Long, high, narrow counter used for serving, often accompanied by tall stools.",
    "arcade machine": "Large, upright wooden cabinet containing a monitor and colorful controls, usually decorated with bright graphics.",
    "hovel": "Small, simple, and often dilapidated dwelling made of rough wood, stone, or mud.",
    "bus": "Large, long rectangular vehicle with many windows and large wheels, used for transporting many people.",
    "towel": "Rectangular piece of thick, absorbent fabric with a soft, fuzzy texture, often hanging from a rack.",
    "light": "Electrical fixture on a wall or ceiling that emits a bright white or yellow glow when active.",
    "truck": "Large vehicle with a separate cab and a heavy, rectangular storage or cargo area in the back.",
    "tower": "Tall, slender vertical structure, either freestanding or part of a building, rising significantly above its surroundings.",
    "chandelier": "Complex, decorative light fixture with multiple branches and glass or crystal ornaments, hanging from a ceiling.",
    "awning": "A sloping, roof-like cover made of canvas or plastic, extending over a door or window to provide shade.",
    "streetlight": "Tall metal pole with a light fixture hanging at the top, used for illuminating roads and sidewalks.",
    "booth": "Small, enclosed or semi-enclosed compartment, typically containing a table and two benches in a restaurant.",
    "television receiver": "Box-like electronic device with a large, flat rectangular glass screen encased in a plastic or metal frame.",
    "airplane": "Large, metallic aerodynamic vehicle with a long cylindrical body, two expansive wings, and a tail fin.",
    "dirt track": "A narrow, unpaved road or path made of packed brown earth or gravel.",
    "apparel": "Fabric items for covering the human body, such as shirts, pants, and coats, shown in various forms and textures.",
    "pole": "Long, thin, vertical cylinder made of wood, metal, or concrete, used as a support or for carrying wires.",
    "land": "General term for solid ground surface, typically shown as brown soil, green vegetation, or rocky terrain.",
    "bannister": "Handrail supported by vertical balusters, running along the side of a staircase.",
    "escalator": "Motorized staircase with repeating steps that move upward or downward in a continuous loop.",
    "ottoman": "Low, padded upholstered piece of furniture without a back or arms, often used as a footrest seat.",
    "bottle": "Tall, slender container with a narrow neck and a wider base, usually made of transparent or colored glass or plastic.",
    "buffet": "Long, low wooden cabinet with doors and drawers, used for storing and serving food items in a dining room.",
    "poster": "Large, flat rectangular sheet of paper with colorful graphics or text, temporarily attached to a wall surface.",
    "stage": "Raised horizontal platform in a theater or hall, usually made of flat wooden planks.",
    "van": "Box-like vehicle larger than a car, with a vertical rear end and sliding side doors.",
    "ship": "Very large vessel for traveling on the sea, with a complex structure including multiple decks and a massive hull.",
    "fountain": "Structure that sprays water into the air, often with a decorative stone basin below.",
    "conveyer belt": "Long, continuous loop of black rubber or metal that moves objects along a flat path.",
    "canopy": "Overhead roof-like structure, often made of fabric supported by a frame, used to provide shade or shelter.",
    "washer": "Box-like white household appliance with a front or top opening for cleaning fabric items.",
    "plaything": "Any small, colorful object with varied shapes and materials designed for amusement, often found in child-related areas.",
    "swimming pool": "Large, rectangular or curved basin excavated into the ground, filled with clear blue water.",
    "stool": "Simple seat with a flat top and three or four legs, usually without a back or arms.",
    "barrel": "Cylindrical container with bulging sides, typically made of vertical wood slats held together by metal hoops.",
    "basket": "Container woven from thin, flexible organic strips like wicker or straw, often featuring a handle.",
    "waterfall": "Water flowing vertically over a steep rock face, characterized by white foam and vertical streaks of movement.",
    "tent": "Portable shelter made of fabric panels stretched over a lightweight pole frame, usually triangular or dome-shaped.",
    "bag": "Flexible container made of paper, plastic, or fabric with handles, used for carrying items.",
    "minibike": "Small, compact motorized vehicle with two wheels, a motor, and a seat for a single rider.",
    "cradle": "Small, bowl-shaped or rectangular bed for a baby, often with curved rockers at the base.",
    "oven": "Enclosed, box-like compartment in a kitchen used for heating, with a front-opening metal and glass door.",
    "ball": "Perfectly spherical object made of rubber, plastic, or leather, used in various games.",
    "food": "Organically shaped, colorful items of various textures (solid, liquid, grainy) served on plates or in containers.",
    "step": "Small, flat horizontal surface that is part of a staircase or used to raise one's standing level.",
    "tank": "Large container for holding liquids or gases, usually cylindrical or rectangular and made of metal or concrete.",
    "trade name": "Text or logo graphics displayed on a signboard, building, or product to identify a brand.",
    "microwave": "Small, box-like kitchen appliance with a front door and a digital display, used for rapid heating.",
    "pot": "Round, deep container usually made of metal with a handle, used for cooking or holding plants.",
    "animal": "Biological form with a head, torso, and limbs, typically covered in fur, feathers, or scales.",
    "bicycle": "Two-wheeled vehicle with a thin metal frame, a seat, and pedals.",
    "lake": "Large, still body of water surrounded by land, often appearing blue or green with calm surface textures.",
    "dishwasher": "Box-like kitchen appliance with a front-opening door, usually found built into a counter for cleaning plates.",
    "screen": "Flat, rectangular surface for displaying images, typically part of a television, computer, or cinema.",
    "blanket": "Large, rectangular piece of soft, thick fabric used for warmth, often spread over a bed.",
    "sculpture": "Three-dimensional artistic form carved or cast from stone, metal, or clay, usually placed on a pedestal.",
    "hood": "Sloping or curved metal cover placed above a stove to collect steam, or the front part of a car.",
    "sconce": "Small light fixture attached directly to a wall at eye level, often containing a decorative shade.",
    "vase": "Decorative container for flowers, typically tall and slender with a flared or tapered neck, made of glass or ceramic.",
    "traffic light": "Vertical metal box with three circular lights (red, yellow, green) stacked on a pole at a road intersection.",
    "tray": "Flat, shallow rectangular plate with raised edges, used for carrying food or small objects.",
    "ashcan": "Cylindrical container for waste, often made of metal or plastic, sometimes with a lid.",
    "fan": "Mechanical device with rotating blades used for moving air, either as a standalone unit or fixed to a ceiling.",
    "pier": "Long structure built from land out into a body of water, supported by thick pillars.",
    "crt screen": "Older, bulky box-like monitor with a slightly curved glass screen surface.",
    "plate": "Flat, circular dish made of ceramic, glass, or plastic, used for serving food.",
    "monitor": "Flat, rectangular electronic screen used to display output from a computer.",
    "bulletin board": "Rectangular board, usually made of brown cork, used for pinning paper notices.",
    "shower": "Small, enclosed bathroom area with a overhead nozzle that sprays water vertically.",
    "radiator": "Metal heating unit with repeating vertical fins, typically found fixed to a wall near a floor.",
    "glass": "Any transparent, solid, and brittle material shown in objects like windowpanes, bottles, or cups.",
    "clock": "Round or square device with a flat face and rotating hands, used for showing time on a wall or table.",
    "flag": "Rectangular piece of colorful, patterned fabric attached to a vertical pole, often waving in the wind."
}

# 按照 datasets/ade.py 中的顺序排列类别 (ADE20K 150 classes)
ade_class_order = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]

def generate_matrix():
    print("Loading CLIP (ViT-B-32) model via sentence-transformers...")
    # 该模型在内部会自动处理文本到 CLIP Vector 的映射
    model = SentenceTransformer('clip-ViT-B-32')
    
    # 提取描述文本列表
    descriptions = [class_descriptions[name] for name in ade_class_order]
    
    print(f"Encoding {len(descriptions)} visual descriptions...")
    with torch.no_grad():
        # 获取文本向量
        embeddings = model.encode(descriptions, convert_to_tensor=True, show_progress_bar=True)
        # 归一化，使得点积等于余弦相似度
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
    # 计算相似度矩阵 [150, 150]
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    
    # 保存结果
    save_dir = "datasets/ade"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "clip_features.npy"), embeddings.cpu().numpy())
    np.save(os.path.join(save_dir, "clip_similarity_matrix.npy"), similarity_matrix.cpu().numpy())
    
    print("Done!")
    print(f"Embeddings saved to: {os.path.join(save_dir, 'clip_features.npy')} (Shape: {embeddings.shape})")
    print(f"Similarity matrix saved to: {os.path.join(save_dir, 'clip_similarity_matrix.npy')} (Shape: {similarity_matrix.shape})")

if __name__ == "__main__":
    generate_matrix()
