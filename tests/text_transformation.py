from interface.text_transformation import RegexpTextTransformation, TfidfFilterTextTransformation


class TestRegexpTextTransformation:

    def test(self):
        assert RegexpTextTransformation(**{"ab": ""}).transform("abcdefgh_abcd") == "bbcdefgh_bbcd"


class TestTfidfFilterTextTransformation:

    def test(self):
        documets = ["aaaaaa ", "aaaaaa bc", "aaaaaa dc", "bbbbb"]

        transformer = TfidfFilterTextTransformation(quantile=0.5)
        transformer.fit(documets)
        result = [transformer.transform(d) for d in documets]

        assert result == ['aaaaaa', 'aaaaaa', 'aaaaaa', 'bbbbb']
