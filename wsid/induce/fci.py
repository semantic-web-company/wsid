def induce_fca2(texts, forms, th_co_fca=0.02):
    new_texts = [None] * len(texts)
    new_forms = forms[:]
    assert forms

    for i in range(len(texts)):
        tokens = [
            stemmer.stem(i)
            for i in word_tokenize(
                preproc.eliminate_shortwords(
                    preproc.eliminate_symbols(texts[i].lower())
                )
            )
            ]
        new_texts[i] = ' '.join(tokens)
    for i in range(len(forms)):
        if isinstance(forms[i], str):
            stemmed = [stemmer.stem(t)
                       for t in word_tokenize(
                    preproc.eliminate_symbols(forms[i].lower())
                )]
            new_forms[i] = r'\b{}\b'.format(' '.join(stemmed))

    canonical_entity_form = 'ENTITY'
    forms_dict = {form: canonical_entity_form for form in new_forms}

    co_co = cooc.get_co(
        new_texts, w, method='unbiased_dice',
        threshold=th_co,
        proximity_func=lambda x: (w - abs(x) + 1) / w,
        forms_dict=forms_dict
    )
    entity_co = co_co[canonical_entity_form]
    if canonical_entity_form in entity_co:
        del entity_co[canonical_entity_form]
    # co_co_values = list(entity_co.values())
    # co_co_th = np.mean(co_co_values)
    # entity_cos_filtered = [co for co in entity_co
    #                        if entity_co[co] >= co_co_th]
    # co_co_E = [(co1, co2, co_co[co1][co2])
    #            for co1 in co_co
    #            for co2 in co_co[co1]
    #            if co_co[co1][co2] > co_co_th]

    # texts context
    objs = []
    table = []
    top_text_cos = set()
    for i in range(len(new_texts)):
        for form in forms_dict:
            subed_text = re.sub(form, forms_dict[form], new_texts[i],
                                flags=re.I)
        text_cos, _ = cooc.get_relevant_tokens(
            subed_text, w, entity=canonical_entity_form
        )
        top_text_cos |= set(sorted(text_cos,
                                   key=lambda x: entity_co[
                                       x] if x in entity_co else -1,
                                   reverse=True)[:5])
    co_co_values = list(entity_co.values())
    co_co_th = np.mean(co_co_values)
    entity_cos_filtered = [co for co in entity_co
                           if entity_co[co] >= co_co_th]
    entity_cos_filtered = list(set(entity_cos_filtered) | top_text_cos)
    for i in range(len(new_texts)):
        objs.append(i)
        for form in forms_dict:
            subed_text = re.sub(form, forms_dict[form], new_texts[i],
                                flags=re.I)
        text_cos, _ = cooc.get_relevant_tokens(
            subed_text, w, entity=canonical_entity_form
        )
        row = [token in text_cos for token in entity_cos_filtered]
        table.append(row)
    cxt_texts = fca.Context(table, objs, entity_cos_filtered)
    print(len(entity_co), '  Relevant cos', len(entity_cos_filtered))
    print('Texts cxt density: ',
          sum(sum(row) for row in table) / (len(table) * len(table[0])))

    # # Choose top concepts
    # topcos = get_top_cos(inf_entity_cos, entity_co, cxt_texts)
    # for co in set(cxt_texts.attributes) - topcos:
    #     cxt_texts.delete_attribute_by_name(co)
    # print('Cooccurrences left: {}'.format(len(topcos)))

    # co context
    objs = list(entity_cos_filtered)
    atts = list(entity_cos_filtered)
    table = []
    for i in range(len(objs)):
        obj = objs[i]
        row = [False] * i + [(att in co_co[obj] and co_co[obj][att] > co_co_th)
                             for att in atts[i:]]
        table.append(row)
    cxt_co = fca.Context(table, objs, atts)
    print('Co co cxt density: ',
          sum(sum(row) for row in table) / (len(table) * len(table[0])))

    # choose senses
    senses = []
    atts_used = set()
    cpt_size = 2
    while True:
        cpts = cxt_co.get_concepts()
        cpt = max(cpts, key=lambda x: (len(x.intent) - 1) * (len(x.extent) - 1))
        cpt_size = (len(cpt.intent) - 1) * (len(cpt.extent) - 1)
        new_sense = cpt.intent | cpt.extent

        foo = []
        for obj in cxt_texts.objects:
            obj_intent = cxt_texts.get_object_intent(obj) - atts_used
            obj_sense_sim = (len(new_sense & obj_intent) /
                             len(obj_intent))
            foo.append((obj_sense_sim, obj, len(new_sense), len(obj_intent)))
        print(foo)
        print([(x[1], x[0]) for ind, x in enumerate(foo) if x[0] >= 0.4])
        senses.append(new_sense)
        atts_used |= new_sense
        for co in new_sense:
            cxt_co.delete_attribute(co)
            cxt_co.delete_object(co)
        # print(cpt_size)
        # print(len(cpt.intent), len(cpt.extent), len(cpt.intent & cpt.extent))
        print(list(cpt.intent | cpt.extent))
    senses.pop()
    print(len(table), len(atts), len(table[0]))
    assert 0

    # doc membership: meet / union

    senses = []
    used_descriptors = set()
    while cpts and cpt_size > size_threshold / 10:
        new_objs = cpt.extent - used_descriptors
        new_atts = cpt.intent - used_descriptors
        novelty = ((len(new_atts) * len(new_objs)) / cpt_size)

        print('Concept: {cpt.extent}, {cpt.intent}'.format(cpt=cpt))
        print('Novelty: {}, size: {}'.format(novelty, cpt_size))
        for text in cxt_texts.objects:
            text_intent = cxt_texts.get_object_intent(text)
            obj_overlap = len(text_intent & cpt.extent) / len(text_intent)
            att_overlap = len(text_intent & cpt.intent) / len(text_intent)
            if text_intent and obj_overlap > 0.1 and att_overlap > 0.1:
                print(text)

        for i in range(len(senses)):
            sense = senses[i]
            cpt_sense_uniqueness = (
            len(cpt.intent - (sense.intent | sense.extent)) *
            len(cpt.extent - (sense.intent | sense.extent)))
            sense_overlap = cpt_size - cpt_sense_uniqueness
            print('Sense overlap: {}'.format(sense_overlap))
            if sense_overlap / cpt_size >= 0.5:
                sense.intent |= cpt.intent
                sense.extent |= cpt.extent
                used_descriptors |= cpt.intent | cpt.extent
                break

        if novelty > 0.75:
            senses.append(cpt)
            used_descriptors |= cpt.intent | cpt.extent

        next_max_cpt = max(
            cpts,
            key=lambda x: len(x.extent - used_descriptors) * len(
                x.intent - used_descriptors)
        )
        cpt = cpts.pop(cpts.index(next_max_cpt))
        cpt_size = len(cpt.intent) * len(cpt.extent)

    ans = []
    covered_texts = set()
    for sense in senses:
        # if len(sense.intent) <= 1 or len(sense.extent) <= 1:
        #     continue
        sense_texts = []
        for text in cxt_texts.objects:
            text_intent = cxt_texts.get_object_intent(text)
            obj_overlap = len(text_intent & sense.extent) / len(sense.extent)
            att_overlap = len(text_intent & sense.intent) / len(sense.intent)
            if text_intent and obj_overlap > 0 and att_overlap > 0:
                sense_texts.append(text)
                # if text_intent and len(text_intent & (sense.intent |
                # sense.extent)) / len(text_intent) > 0.5:
        if len(sense_texts) >= 3:
            ans.append([sense_texts, sense])
            covered_texts |= set(sense_texts)

    print('Finished.\n\nSenses: {}\nTexts:{}'.format(len(senses),
                                                     len(covered_texts)))
    return ans