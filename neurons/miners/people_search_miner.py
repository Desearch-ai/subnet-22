import os
import bittensor as bt
from datura.protocol import PeopleSearchSynapse, PeopleSearchResult
from neurons.validators.apify.linkedin_scraper_actor import LinkedinScraperActor


SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError(
        "Please set the SERPAPI_API_KEY environment variable. See here: https://github.com/Datura-ai/desearch/blob/main/docs/env_variables.md"
    )


class PeopleSearchMiner:
    def __init__(self, miner: any):
        self.miner = miner
        self.linkedin_scraper_actor = LinkedinScraperActor()

    async def search(self, synapse: PeopleSearchSynapse):
        # Extract the query from the synapse
        query = synapse.query

        # Log the mock search execution
        bt.logging.info(f"Executing people search with query: {query}")

        links = [
            "https://uk.linkedin.com/in/ethanghoreishi",
            "https://uk.linkedin.com/in/vitali-avagyan-phd-a1566234",
            "https://uk.linkedin.com/in/olly-styles-090437132",
            "https://uk.linkedin.com/in/nvedd",
            "https://www.linkedin.com/in/jean-kaddour-344837267",
        ]

        # profiles = await self.linkedin_scraper_actor.get_profiles(links)

        synapse.results = [
            {
                "link": "https://uk.linkedin.com/in/ethanghoreishi",
                "first_name": "Ethan",
                "last_name": "Ghoreishi",
                "full_name": "Ethan Ghoreishi",
                "title": "Co-Founder & CTO at Teamflow",
                "summary": "Strong background in data science and entrepreneurship with a track record of building innovative data-driven solutions to a variety of business problems.\n\nExpert using machine/deep learning and experience managing and maintaining large scale data sets on cloud platforms.\n\nPhD in Informatics, 6 publications in the AI space in top-tier journals and conferences.\n\nMain Areas of Expertise:\nMachine Learning • Deep Neural Networks • Natural Language Processing (NLP) • Mathematical Optimisation • Probability & Statistics • Data Modelling • Agile Software Development  • Market Research • Customer Development\n",
                "avatar": "https://media.licdn.com/dms/image/v2/C5603AQHaxnK2fuc-Fg/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1551908282179?e=1747872000&v=beta&t=kLMEN6u2E1QpPlPAtYDQwobUVXZ1abUD5hC6U9b4j1E",
                "experiences": [
                    {
                        "company_id": "11811849",
                        "company_link": "https://www.linkedin.com/company/11811849/",
                        "title": "Co-Founder",
                        "subtitle": "Teamflow",
                        "caption": "2018 - Present · 7 yrs 3 mos",
                        "metadata": None,
                    },
                    {
                        "company_id": "2340783",
                        "company_link": "https://www.linkedin.com/company/2340783/",
                        "title": "Entrepreneur In Residence",
                        "subtitle": "Entrepreneur First",
                        "caption": "Oct 2017 - Oct 2018 · 1 yr 1 mo",
                        "metadata": "London, United Kingdom",
                    },
                    {
                        "company_id": "10958505",
                        "company_link": "https://www.linkedin.com/company/10958505/",
                        "title": "Co-Founder",
                        "subtitle": "Intuition Tech",
                        "caption": "Sep 2016 - Nov 2017 · 1 yr 3 mos",
                        "metadata": "London, United Kingdom",
                    },
                    {
                        "company_id": "11236367",
                        "company_link": "https://www.linkedin.com/company/11236367/",
                        "title": "Founder & Technical Lead",
                        "subtitle": "Orcai",
                        "caption": "Apr 2017 - Sep 2017 · 6 mos",
                        "metadata": "London, United Kingdom",
                    },
                    {
                        "company_id": "10402212",
                        "company_link": "https://www.linkedin.com/company/10402212/",
                        "title": "Data Science Fellow",
                        "subtitle": "Science to Data Science (S2DS)",
                        "caption": "Aug 2016 - Sep 2016 · 2 mos",
                        "metadata": "London, United Kingdom",
                    },
                ],
                "educations": [
                    {
                        "company_id": "7198",
                        "company_link": "https://www.linkedin.com/company/7198/",
                        "title": "King's College London",
                        "subtitle": "Doctor of Philosophy (Ph.D.), Informatics ",
                        "caption": "2013 - 2016",
                    },
                    {
                        "company_id": None,
                        "company_link": "https://www.linkedin.com/search/results/all/?keywords=Birmingham+University",
                        "title": "Birmingham University",
                        "subtitle": "Master of Science - MS, Data Networks and Security",
                        "caption": "2012 - 2013",
                    },
                ],
                "languages": [],
                "relevance_summary": None,
                "criteria_summary": [],
                "extra_information": None,
            },
            {
                "link": "https://uk.linkedin.com/in/vitali-avagyan-phd-a1566234",
                "first_name": "Vitali",
                "last_name": "Avagyan, PhD",
                "full_name": "Vitali Avagyan, PhD",
                "title": "Co-Founder/CTO @Caraxel AI | I get the job done",
                "summary": "Vitali is an experienced Machine Learning Engineer proficient in statistical analysis and software engineering. \n\nHe's skilled in programming languages such as Python, R and SQL, and has applied these in the energy, finance, and logistics sectors. \n\nHis expertise extends to Linux, cloud computing, distributed machine learning, and deep-learning frameworks. \n\nVitali is also recognized for his excellent communication and impactful teaching style.",
                "avatar": "https://media.licdn.com/dms/image/v2/C5603AQGNjoZx6tBHnA/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1516564044098?e=1747872000&v=beta&t=sOBsGNu9MU57c3ydqXCT9EACoR7FIa9R_GF7mFZxwwQ",
                "experiences": [
                    {
                        "company_id": "105578031",
                        "company_link": "https://www.linkedin.com/company/105578031/",
                        "title": "CTO  & Co-founder",
                        "subtitle": "Caraxel AI · Full-time",
                        "caption": "Jan 2025 - Present · 3 mos",
                        "metadata": "Hybrid",
                    },
                    {
                        "company_id": "97848618",
                        "company_link": "https://www.linkedin.com/company/97848618/",
                        "title": "Co-Founder/CTO",
                        "subtitle": "MyMagic AI · Full-time",
                        "caption": "Jul 2023 - Jan 2025 · 1 yr 7 mos",
                        "metadata": "San Francisco Bay Area · Hybrid",
                    },
                    {
                        "company_id": "583531",
                        "company_link": "https://www.linkedin.com/company/583531/",
                        "title": "Visiting Professor of Artificial Intelligence",
                        "subtitle": "Yerevan State University · Seasonal",
                        "caption": "Sep 2023 - Jan 2024 · 5 mos",
                        "metadata": "Yerevan, Armenia · On-site",
                    },
                    {
                        "company_id": "10971923",
                        "company_link": "https://www.linkedin.com/company/10971923/",
                        "title": "AI Engineer/ Senior Data Scientist",
                        "subtitle": "TurinTech AI · Full-time",
                        "caption": "Dec 2021 - Jun 2023 · 1 yr 7 mos",
                        "metadata": "London, England, United Kingdom",
                    },
                    {
                        "company_id": "4156",
                        "company_link": "https://www.linkedin.com/company/4156/",
                        "title": "Air Products",
                        "subtitle": "2 yrs 10 mos",
                        "caption": None,
                        "metadata": None,
                    },
                ],
                "educations": [
                    {
                        "company_id": "5106",
                        "company_link": "https://www.linkedin.com/company/5106/",
                        "title": "Imperial College London",
                        "subtitle": "Doctor of Philosophy (Ph.D.), Optimisation, Economics, AI, Energy ",
                        "caption": "2012 - 2016",
                    },
                    {
                        "company_id": None,
                        "company_link": "https://www.linkedin.com/search/results/all/?keywords=American+University+of+Armenia",
                        "title": "American University of Armenia",
                        "subtitle": "Master, Industrial Engineering",
                        "caption": "2010 - 2012",
                    },
                ],
                "languages": [
                    {"title": "Armenian", "caption": "Native or bilingual proficiency"},
                    {"title": "English", "caption": "Full professional proficiency"},
                ],
                "relevance_summary": None,
                "criteria_summary": [],
                "extra_information": None,
            },
            {
                "link": "https://uk.linkedin.com/in/nvedd",
                "first_name": "Nihir",
                "last_name": "Vedd",
                "full_name": "Nihir Vedd",
                "title": "CTO | Generative AI PhD & Lecturer",
                "summary": "Currently building Glimpses.ai which is a fashion personalisation & search engine.\n\nOtherwise... Published generative AI author with a focus on NLP and multi-modal AI. PhD at Imperial College London. Start-up founder.\n\nMy day-to-day on my PhD involved me working on the same technology which powers ChatGPT and Google search. I am also a seasoned full-stack developer, second time founder, and guest lecturer in NLP at Imperial College.",
                "avatar": "https://media.licdn.com/dms/image/v2/D4E03AQGts4Ee6keTXQ/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1689164966027?e=1747872000&v=beta&t=-h4B8zRU-QceufLi1HdJQFLq7tHrxGh9tS9Rg9eNHac",
                "experiences": [
                    {
                        "company_id": None,
                        "company_link": "https://www.linkedin.com/search/results/all/?keywords=Glimpses+AI",
                        "title": "Co-Founder",
                        "subtitle": "Glimpses AI · Full-time",
                        "caption": "Dec 2023 - Present · 1 yr 4 mos",
                        "metadata": None,
                    },
                    {
                        "company_id": "2340783",
                        "company_link": "https://www.linkedin.com/company/2340783/",
                        "title": "Entrepreneur in Residence",
                        "subtitle": "Entrepreneur First · Full-time",
                        "caption": "Apr 2023 - Jul 2023 · 4 mos",
                        "metadata": "London, England, United Kingdom · On-site",
                    },
                    {
                        "company_id": None,
                        "company_link": "https://www.linkedin.com/search/results/all/?keywords=feather",
                        "title": "Founder",
                        "subtitle": "feather · Full-time",
                        "caption": "Apr 2021 - Apr 2022 · 1 yr 1 mo",
                        "metadata": "London, England, United Kingdom",
                    },
                    {
                        "company_id": "18690627",
                        "company_link": "https://www.linkedin.com/company/18690627/",
                        "title": "Lead Instructor/Syllabus Lead",
                        "subtitle": "The AI Core · Full-time",
                        "caption": "Jan 2020 - Jan 2021 · 1 yr 1 mo",
                        "metadata": None,
                    },
                    {
                        "company_id": "4856",
                        "company_link": "https://www.linkedin.com/company/4856/",
                        "title": "Frontend Engineer",
                        "subtitle": "EF Education First · Contract",
                        "caption": "Nov 2018 - Aug 2019 · 10 mos",
                        "metadata": "London, United Kingdom",
                    },
                ],
                "educations": [
                    {
                        "company_id": "5106",
                        "company_link": "https://www.linkedin.com/company/5106/",
                        "title": "Imperial College London",
                        "subtitle": "Doctor of Philosophy - PhD, Natural Language Processing (NLP)",
                        "caption": "2019 - 2023",
                    },
                    {
                        "company_id": "10250",
                        "company_link": "https://www.linkedin.com/company/10250/",
                        "title": "Lancaster University",
                        "subtitle": "Master's degree, Data Processing",
                        "caption": "2017 - 2018",
                    },
                ],
                "languages": [],
                "relevance_summary": None,
                "criteria_summary": [],
                "extra_information": None,
            },
            {
                "link": "https://www.linkedin.com/in/jean-kaddour-344837267",
                "first_name": "Jean",
                "last_name": "Kaddour",
                "full_name": "Jean Kaddour",
                "title": "PySpur.dev | PhD in LLMs @UCL",
                "summary": "https://www.jeankaddour.com/",
                "avatar": "https://media.licdn.com/dms/image/v2/D4E03AQGQWsTlp4ioVg/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1677276387013?e=1747872000&v=beta&t=LgL_E6wUZ8iGVhOus3bWBuaicyCoINmQ1TpKT1EI5mc",
                "experiences": [
                    {
                        "company_id": "100110157",
                        "company_link": "https://www.linkedin.com/company/100110157/",
                        "title": "Co-Founder",
                        "subtitle": "PySpur · Full-time",
                        "caption": "Nov 2024 - Present · 5 mos",
                        "metadata": "London, England, Vereinigtes Königreich · On-site",
                    },
                    {
                        "company_id": "98133663",
                        "company_link": "https://www.linkedin.com/company/98133663/",
                        "title": "Research Intern",
                        "subtitle": "Reka AI · Internship",
                        "caption": "Oct 2022 - Jan 2023 · 4 mos",
                        "metadata": "London, England, United Kingdom · On-site",
                    },
                    {
                        "company_id": "1784",
                        "company_link": "https://www.linkedin.com/company/1784/",
                        "title": "Visiting Data Scientist",
                        "subtitle": "Boston Consulting Group (BCG) · Internship",
                        "caption": "Sep 2019 - Dec 2019 · 4 mos",
                        "metadata": None,
                    },
                ],
                "educations": [
                    {
                        "company_id": "4171",
                        "company_link": "https://www.linkedin.com/company/4171/",
                        "title": "UCL",
                        "subtitle": "PhD, Machine Learning",
                        "caption": "Sep 2020 - 2023",
                    },
                    {
                        "company_id": "5106",
                        "company_link": "https://www.linkedin.com/company/5106/",
                        "title": "Imperial College London",
                        "subtitle": "Master of Science - MS, Advanced Computing",
                        "caption": None,
                    },
                ],
                "languages": [],
                "relevance_summary": None,
                "criteria_summary": [],
                "extra_information": None,
            },
        ]

        return synapse
