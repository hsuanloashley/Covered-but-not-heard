# Outlet type mapping. PLEASE REVIEW (exported to revision_outputs/outlet_type_mapping.csv).
# broadsheet = legacy print newspaper (national or metro); cable = TV/cable/broadcast news digital arms;
# wire = wire/news agency; digital = digital-native (incl. partisan digital, magazines' online arms coded 'magazine').
BROADSHEET={'seattletimes.com','washingtonpost.com','nytimes.com','sfgate.com','startribune.com','wsj.com','newsday.com','staradvertiser.com','nypost.com','latimes.com','sandiegouniontribune.com','arkansasonline.com','stltoday.com','usatoday.com','azcentral.com','mercurynews.com','centralmaine.com','ajc.com','chicagotribune.com','ocregister.com','baltimoresun.com','providencejournal.com','dispatch.com','denverpost.com','sun-sentinel.com','pilotonline.com','bostonglobe.com','twincities.com','sacbee.com','kansascity.com','orlandosentinel.com','cleveland.com','oregonlive.com','miamiherald.com','courant.com','freep.com','dallasnews.com','buffalonews.com','tampabay.com','jsonline.com','cincinnati.com','theguardian.com','csmonitor.com','stripes.com','barrons.com','investors.com','marketwatch.com'}
CABLE={'abcnews.go.com','cnn.com','foxnews.com','cbsnews.com','oann.com','npr.org','nbcnews.com','cnbc.com','foxbusiness.com','newsmax.com','pri.org','espn.com','bloomberg.com'}
WIRE={'reuters.com','upi.com'}
MAGAZINE={'newsweek.com','forbes.com','time.com','usnews.com','fortune.com','theweek.com','nationalreview.com','theatlantic.com','wired.com','newyorker.com','vanityfair.com','thenation.com','motherjones.com','scientificamerican.com','people.com','variety.com','hollywoodreporter.com','spectator.org','ncregister.com','livescience.com'}
# everything else -> digital-native
def outlet_type(m):
    if m in BROADSHEET: return 'broadsheet'
    if m in CABLE: return 'cable/broadcast'
    if m in WIRE: return 'wire'
    if m in MAGAZINE: return 'magazine'
    return 'digital-native'
