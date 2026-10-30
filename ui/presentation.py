"""Atlas presentation only: no employee data, model calls, or business operations."""
import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / 'assets'


@st.cache_data(max_entries=8, show_spinner=False)
def _media_uri(filename, modified_ns):
    """Cache only public, bundled media. No employee/session content is cached."""
    mime = 'video/mp4' if filename.endswith('.mp4') else 'image/jpeg'
    return f'data:{mime};base64,' + base64.b64encode((ASSETS / filename).read_bytes()).decode()


def _asset(filename):
    return _media_uri(filename, (ASSETS / filename).stat().st_mtime_ns)


def render_masthead():
    st.html('''<header class="atlas-masthead">
      <a class="atlas-brand" href="/" target="_self" aria-label="Atlas home">
        <span class="atlas-symbol" aria-hidden="true">a</span>atlas<span class="atlas-brand-divider">/</span><span class="atlas-company">SAMSARA</span>
      </a>
      <nav class="atlas-top-nav" aria-label="Employee resources">
        <a href="/policies" target="_self">Policy library <span aria-hidden="true">↗</span></a>
        <a class="atlas-workspace-link" href="#atlas-workspace">Your workspace <span aria-hidden="true">↘</span></a>
      </nav>
    </header>''')


def render_logistics_header():
    """Two licensed, locally hosted five-second loops with native pause controls."""
    reduced = st.session_state.get('reduce_motion', False)
    cards = []
    for slug, number, label, description in (
        ('construction', '01', 'Building what comes next', 'Aerial construction footage of an excavator at work'),
        ('shipping', '02', 'Keeping the world moving', 'Aerial shipping footage of containers, cranes and a cargo ship'),
    ):
        poster = _asset(slug + '-hero.jpg')
        video = '' if reduced else (
            f'<video class="atlas-header-video" aria-label="{description}" '
            f'autoplay muted loop playsinline controls preload="metadata" poster="{poster}">'
            f'<source src="{_asset(slug + "-hero.mp4")}" type="video/mp4"></video>')
        cards.append(f'''<figure class="atlas-film atlas-film-{slug}">
          <div class="atlas-film-top"><span class="atlas-film-number">{number}</span><span>{slug.upper()}</span><span class="atlas-film-mark" aria-hidden="true">↗</span></div>
          <div class="atlas-film-screen">{video}<img class="atlas-header-poster" src="{poster}" alt="{description}"></div>
          <figcaption>{label}<span aria-hidden="true">↗</span></figcaption>
        </figure>''')
    st.html(f'''<style>.atlas-header-poster{{display:{'block' if reduced else 'none'}}}
      @media(prefers-reduced-motion:reduce){{.atlas-header-video{{display:none}}.atlas-header-poster{{display:block}}}}
      </style><section class="atlas-films" aria-label="The operations our people power">
      <div class="atlas-connector" aria-hidden="true"></div>{cards[0]}
      <div class="atlas-films-center"><span class="atlas-center-icon" aria-hidden="true">✳</span>
        <span>People.<br>Possibility.<br>Progress.</span><small>CONNECTED BY YOU</small></div>{cards[1]}</section>''')


def render_workspace_intro():
    st.html('''<div class="atlas-workspace-intro" id="atlas-workspace">
      <span class="atlas-eyebrow">A LITTLE CLARITY GOES A LONG WAY</span>
      <span>01 &nbsp; Ask in your own words <b aria-hidden="true">→</b> 02 &nbsp; Explore the evidence <b aria-hidden="true">→</b> 03 &nbsp; Review with HR</span>
    </div>''')


def apply_presentation():
    reduced = st.session_state.get('reduce_motion', False)
    st.html('''<style>
    :root{--atlas-blue:#304bf2;--atlas-ink:#20211d;--atlas-muted:#5c6057;--atlas-paper:#f2f2e8;--atlas-line:#d8dacf}
    .stApp{background:var(--atlas-paper);color:var(--atlas-ink)}
    [data-testid="stMainBlockContainer"]{max-width:1180px;margin:auto;padding-left:3rem;padding-right:3rem}
    [data-testid="stHeader"]{background:var(--atlas-paper)}
    p,li{line-height:1.65}
    h1{font-family:Arial,Helvetica,sans-serif!important;font-size:clamp(2.1rem,4.8vw,4.3rem)!important;font-weight:500!important;letter-spacing:-.055em!important;line-height:1.07!important;max-width:1030px;padding-top:1.4rem!important;padding-bottom:1rem!important}
    h2,h3{letter-spacing:-.035em!important;font-weight:500!important}
    [data-testid="stCaptionContainer"]{opacity:1!important}
    [data-testid="stCaptionContainer"] p{color:var(--atlas-muted)!important}
    [data-testid="stText"]{white-space:pre-wrap;overflow-wrap:anywhere}
    [data-testid="stSidebar"]{background:#e9ebdf;border-right:1px solid var(--atlas-line)}
    [data-testid="stVerticalBlockBorderWrapper"]{border-radius:16px}
    [data-testid="stForm"]{background:#fcfcf5;border:1px solid #cfd3c2;border-radius:18px;padding:1.5rem;box-shadow:0 12px 35px #343b2510}
    button{min-height:44px;border-radius:9px!important;transition:background-color 120ms ease,border-color 120ms ease!important}
    button[kind="primary"],button[kind="primaryFormSubmit"]{background:var(--atlas-blue)!important;border-color:var(--atlas-blue)!important;color:#fff!important}
    textarea{font-size:1.1rem!important;line-height:1.6!important}
    :is(button,a,input,textarea,select):focus-visible{outline:3px solid var(--atlas-blue)!important;outline-offset:4px!important}
    .atlas-masthead{display:flex;align-items:center;justify-content:space-between;gap:1.25rem;padding:.4rem 0 1.5rem;border-bottom:1px solid var(--atlas-line);margin-bottom:1.6rem}
    .atlas-brand{display:flex;align-items:center;gap:.55rem;color:var(--atlas-ink)!important;text-decoration:none!important;font:500 2rem/1 Arial,sans-serif;letter-spacing:-.07em}
    .atlas-symbol{width:31px;height:31px;background:var(--atlas-blue);color:#fff;display:grid;place-items:center;border-radius:7px;font:500 31px/1 Georgia,serif;letter-spacing:-.06em;padding-bottom:4px}
    .atlas-brand-divider{font:300 1.35rem/1 Arial,sans-serif;color:#a1a598;letter-spacing:0;margin:0 .3rem}
    .atlas-company{font:600 .65rem/1.2 Arial,sans-serif;letter-spacing:.14em}
    .atlas-top-nav{display:flex;align-items:center;gap:1.7rem;font-size:.88rem}
    .atlas-top-nav a{color:#45483f;text-decoration:none;min-height:44px;display:inline-flex;gap:.65rem;align-items:center}
    .atlas-top-nav a:hover{color:var(--atlas-blue)}
    .atlas-top-nav .atlas-workspace-link{background:var(--atlas-blue);color:white;border-radius:8px;padding:.2rem 1rem}
    .atlas-top-nav .atlas-workspace-link:hover{background:#233cce;color:#fff}
    .atlas-kicker,.atlas-eyebrow{font:500 .65rem/1.5 ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;color:#59624e}
    .atlas-kicker{margin:1rem 0 -.5rem;color:var(--atlas-blue)}
    .atlas-lead{font-size:1.1rem;color:#555c50;max-width:42rem;margin:0 0 1.8rem;line-height:1.65}
    .atlas-lead strong{font-weight:400}
    .atlas-films{display:grid;grid-template-columns:1fr .48fr 1fr;gap:1.5rem;align-items:center;position:relative;padding:1rem 0 .75rem}
    .atlas-connector{position:absolute;left:0;right:0;top:52%;height:1px;background:var(--atlas-blue);opacity:.55}
    .atlas-film{margin:0;position:relative;background:#fafbf4;border:1px solid #d4d8c9;border-radius:14px;overflow:hidden;box-shadow:0 16px 32px #303c2414}
    .atlas-film-top{display:flex;align-items:center;gap:.6rem;padding:.65rem .85rem;font:500 .6rem/1.5 ui-monospace,SFMono-Regular,monospace;letter-spacing:.08em;color:#454d3e}
    .atlas-film-number{color:var(--atlas-blue);background:#e7eaff;border-radius:4px;padding:2px 5px;letter-spacing:0}
    .atlas-film-mark{margin-left:auto;color:var(--atlas-blue);font-size:.8rem}
    .atlas-film-screen{aspect-ratio:5/3;background:#e4e7db;overflow:hidden}
    .atlas-film-screen video,.atlas-film-screen img{width:100%;height:100%;object-fit:cover;display:block}
    .atlas-film-screen .atlas-header-poster{display:none}
    .atlas-film figcaption{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.8rem .85rem;font-size:.8rem;color:#404938}
    .atlas-film figcaption span{color:var(--atlas-blue)}
    .atlas-films-center{position:relative;text-align:center;background:var(--atlas-paper);padding:1rem 0;display:flex;flex-direction:column;align-items:center;gap:.7rem}
    .atlas-center-icon{color:var(--atlas-blue);font-size:2.8rem;line-height:1}
    .atlas-films-center>span:nth-child(2){font-size:1.15rem;letter-spacing:-.035em;line-height:1.12}
    .atlas-films-center small{font:500 .5rem/1.5 ui-monospace,SFMono-Regular,monospace;letter-spacing:.12em;color:#68735b}
    .atlas-workspace-intro{border-top:1px solid #c9cebd;border-bottom:1px solid #c9cebd;padding:1.1rem 0;margin:.3rem 0 1.7rem;display:flex;justify-content:space-between;gap:1.25rem;align-items:center;scroll-margin-top:5rem}
    .atlas-workspace-intro>span:last-child{font-size:.7rem;color:#5b6252;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap}
    .atlas-workspace-intro b{color:var(--atlas-blue);font-weight:400}
    .atlas-rule{height:1px;background:var(--atlas-line);margin:1.5rem 0}
    @media(max-width:760px){[data-testid="stMainBlockContainer"]{padding-left:1.2rem;padding-right:1.2rem}.atlas-films{grid-template-columns:1fr 1fr;gap:.8rem}.atlas-films-center{display:none}.atlas-masthead{gap:.7rem}.atlas-top-nav{gap:.7rem;font-size:.75rem}.atlas-top-nav .atlas-workspace-link{padding:.15rem .65rem}.atlas-company,.atlas-brand-divider{display:none}.atlas-workspace-intro{align-items:flex-start;flex-direction:column;gap:.75rem}.atlas-film-top{font-size:.5rem;padding:.5rem}.atlas-film figcaption{padding:.65rem .5rem;font-size:.68rem}.atlas-lead{font-size:1rem}h1{font-size:2.6rem!important}}
    @media(max-width:420px){.atlas-top-nav>a:first-child{display:none}.atlas-film figcaption{font-size:.61rem}.atlas-film-top{letter-spacing:.02em}.atlas-film-number{padding:1px 3px}h1{font-size:2.15rem!important}[data-testid="stForm"]{padding:1rem}}
    @media(prefers-reduced-motion:reduce){button,*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}.atlas-film-screen .atlas-header-video{display:none}.atlas-film-screen .atlas-header-poster{display:block}}
    </style>''')
    if reduced:
        st.html('<style>button,*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}.atlas-film-screen .atlas-header-poster{display:block}</style>')


def render_answer(outputs):
    st.subheader('Your answer draft')
    st.warning('Draft only: HR review is required. Workflow completion does not mean the answer is approved.')
    answer, sources = st.columns([3, 2], gap='large')
    with answer:
        with st.container(border=True):
            st.text(str(outputs.get('subject', '')))
            st.write(outputs.get('body', ''))
    with sources:
        st.markdown('#### Supporting policy')
        quotes = outputs.get('cited_snippets', [])
        evidence = outputs.get('citation_evidence', [])
        if not quotes:
            st.caption('No policy citation is available for this draft. Ask HR to verify the guidance.')
        for index, quote in enumerate(quotes):
            with st.container(border=True):
                source = next((e.get('source_file') for e in evidence if e.get('quote') == quote), None)
                st.caption(source or f'Policy excerpt {index + 1}')
                st.text(quote)
        st.caption('A citation identifies source text; it does not confirm eligibility or approval.')
