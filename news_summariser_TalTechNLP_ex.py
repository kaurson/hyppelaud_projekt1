from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("TalTechNLP/mBART-ERRnews")
model = AutoModelForSeq2SeqLM.from_pretrained("TalTechNLP/mBART-ERRnews")

text = """
       "Eestis tegutsevate pankade kasumid kerkisid tänavu esimesel poolaastal pea samale tasemele kogu möödunud aasta kasumitega.

Suurima Eestis tegutseva panga Swedbanki selle aasta esimese kuue kuu puhaskasum oli 214 miljonit eurot, mis oli möödunud aasta sama perioodi kasumist 148 miljonit eurot suurem. Ainuüksi intressitulu kasvas mullu sama ajaga võrreldes 150 miljoni võrra, selle kasvu toetasid suurem laenuportfell ning euribori toel kasvanud intressid.

Võrdluseks: panga kogu eelmise aasta kasum oli 224 miljonit eurot.

Panga tulud kasvasid 192 miljoni euro võrra, seda peamiselt suuremate intressitulude toel, mille põhjustasid peamiselt suuremad tulud Euroopa Keskpanga tõstetud intressimäärade tõttu, teatas Swedbank pressiteate vahendusel.

Swedbanki eraklientide laenumahud kasvasid aastaga 5,5 protsenti ja äriklientide omad 4 protsenti. Hoiused kasvasid ühe protsendi võrra. Aasta esimese kuue kuuga väljastas Swedbank eraisikutele uusi laene kogusummas 521 miljonit eurot ja ettevõtetele 800 miljonit eurot.

Klientide suurem kaartide käive ja ettevõtete suurem laenulimiidi kasutus kasvatas puhast teenustasutulu ühe miljoni euro võrra. Väiksemad maksete ning väärtpaberitega seotud tulud vähendasid kaartide positiivset mõju, märgib pank.

Eeldatavad krediidikahjud tänavu esimese kuue kuuga ulatusid –4,5 miljoni euroni; 2022. aasta samal ajal oli vastav näitaja –0,8 miljonit eurot.

"Eesti majandust mõjutavad jätkuvalt suurenev ebakindlus ja kõrgemad elamiskulud, mõnede toodete ja teenuste tarbimisel ollakse ettevaatlikumad. Samas, meie kliendiportfelli pole see veel mõjutama hakanud," ütles Swedbanki juht Olavi Lepp.


Swedbank Eesti esimese kvartali kasum kasvas 105 miljonile eurole
Puhastulu finantsinstrumentidelt kasvas 5 miljoni euro võrra, tingituna varade ümberhindluse realiseerimata kasumist varahalduse ning kindlustuse valdkondades, mille positiivset mõju vähendas väiksem tulu valuutakauplemise tehingutest.

Muud tulud kasvasid 36 miljoni euro võrra suurenenud kindlustustulude toel.

Ettevõtte kulud kasvasid 13 miljoni euro võrra. Kulude tõusu peamiste põhjustena toob pank välja suuremad personalikulud ning Swedbanki grupist sisse ostetavate teenuste kulude kasvu. Samuti kasvasid digilahendustega seotud kulud ja investeeringud.

LHV aasta esimese kuue kuu puhaskasum oli 68,9 miljonit eurot ehk 42 miljonit eurot rohkem kui mullu samal ajal. Möödunud aastal oli grupi puhaskasum 61,4 miljonit eurot.


Ettevõtete finantstervist peab SEB heaks, ettevõtete hoiused kasvasid aastaga 12,8 protsenti. "Uute investeeringute tegemises ollakse samas ettevaatlikud, sest keskkond on jätkuvalt ebaselge. Siiski oleme suutnud ettevõtete laenuportfelli aastaga suurendada 6 protsenti, mis tähendab, et panga investeeringud Eesti ettevõtetesse küündisid poolaasta seisuga pea 2,9 miljardi euroni," märkis Parik.
"""
inputs = tokenizer(text, return_tensors='pt', max_length=1024)

summary_ids = model.generate(inputs['input_ids'])
summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

print(summary)