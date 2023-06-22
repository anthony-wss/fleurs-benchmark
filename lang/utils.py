import iso639, dataclasses

def get_iso_693_3(lang_id):
    if lang_id == 'jw':
        return 'jav'
    if lang_id == 'iw':
        return 'heb'
    if lang_id == 'zh':
        return 'cmn'
    language = iso639.Language.match(lang_id)
    code = dataclasses.asdict(language)['part3']
    if lang_id == 'mya':
        return 'bur'
    if lang_id == 'tgl':
        return 'fil'
    return code