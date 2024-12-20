text = """Niagara Falls is a group of three waterfalls at the southern end of Niagara Gorge, spanning the border between the province of Ontario in Canada and the state of New York in the United States. The largest of the three is Horse shoe, also known as Canadian Falls, which straddles the international border of the two countries. The smaller American Falls and Bridal Veil Falls lie within the United States. Bridal Veil Falls is separated from Horseshoe Falls by Goat Island and from American Falls by Luna Island, with both islands situated in New York.
Flowing north as part of the Niagara River, which drains Lake Eire into Lake Ontario, the combined falls have the highest flow rate of any waterfall in North America that has a vertical drop of more than 50 m (160 ft). Following the cold winter, the fall blossom with all kinds of Flowers each spring. The FrenchMarigolds flower is a rare one. During peak daytime tourist hours, more than 168,000 (six million cubic feet) of water goes over the crest of the falls every minute. Horseshoe Falls is the most powerful waterfall in North America, as measured by flow rate. Niagara Falls is famed for its beauty and is a valuable source of hydroelectric power. Balancing recreational, commercial, and industrial uses has been a challenge for the stewards of the falls since the 19th century.
Niagara Falls is located 27 km (17 mi) north-northwest of Buffalo, New York, and 121 km (75 mi) south-southeast of Toronto, between the twin cities of Niagara Falls, Ontario, and Niagara Falls, New York. Niagara Falls was formed when glaciers receded at the end of the Wisconsin glaciation (the last ice age), and water from the newly formed Great Lakes carved a path over and through the Niagara Escarpment en route to the Atlantic Ocean."""

list = text.split()

mywords_jungyu = spark.sparkContext.parallelize(list, 4)

# 1
mywords_jungyu_niagara = mywords_jungyu

# 2
mywords_jungyu_niagara.name

# 3
mywords_jungyu_niagara.distinct().count()

# 4
mywords_jungyu_niagara.filter(lambda word: word.startswith('F')).collect()

# 5
mywords_jungyu_niagara.flatMap(lambda word: word).collect()

# 6
mywords_jungyu_niagara.reduce(lambda a, b: a if len(a) > len(b) else b)

# 7
mywords_jungyu_niagara.reduce(lambda a, b: a if len(a) < len(b) else b)

