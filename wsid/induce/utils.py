class InducedSense:
    def __init__(self, hub, score, broaders, cluster):
        """

        :param str hub:
        :param float score:
        :param list[str] broaders:
        :param dict[str, float] cluster:
        """
        self.hub = hub
        self.score = score
        self.broaders = broaders
        self.cluster = cluster
        self._top_indicators = None

    @classmethod
    def hub2sense(cls, hub, cluster):
        hub_head, score, broaders = hub
        sense = InducedSense(
            hub=hub_head, score=score, broaders=broaders, cluster=cluster)
        return sense

    @property
    def top_indicators(self, n=10):
        if self._top_indicators is None:
            _top_indicators = sorted(
                self.cluster.items(), key=lambda x: x[1], reverse=True)
            self._top_indicators = _top_indicators
        return self._top_indicators[:n]

    def __str__(self, verbose=False):
        s = f'InducedSense(hub={self.hub}, score={self.score}, ' \
            f'broaders={self.broaders})'
        if verbose:
            s += '\nTop Indicators:\n'
            s += '\n'.join(f'{i}) {self.top_indicators[i]}'
                           for i in range(10))
        return s


class InducedModel:
    def __init__(self, lemma, pos, senses):
        """

        :param str lemma:
        :param str pos:
        :param list[InducedSense] senses:
        """
        self.lemma = lemma
        self.pos = pos
        self.senses = senses

    def __str__(self, verbose=False):
        s = f'Lemma "{self.lemma}", POS = {self.pos}\n'
        s += '\n\n'.join(f'{i} ' + s.__str__(verbose=verbose)
                         for i, s in enumerate(self.senses))
        return s
