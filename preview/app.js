(() => {
  const { profile, biography, papers, honors, service } = window.ACADEMIC_DATA;
  const sidebarLayout = document.body.dataset.layout === 'b';
  const app = document.getElementById('app');
  const compactViewport = matchMedia('(max-width: 900px)');
  const menuIcon = 'M4 7h16M4 12h16M4 17h16';
  const closeIcon = 'M6 6l12 12M18 6L6 18';
  const sections = ['about', 'publications', 'honors', 'service'];
  let lang = new URLSearchParams(location.search).get('lang') === 'zh' ? 'zh' : 'en';
  let alternatePortrait = false;
  let navigationObserver;
  let figurePreview;

  const labels = {
    en: {
      about: 'About', publications: 'Publications', honors: 'Honors & awards',
      service: 'Academic service', navHonors: 'Honors & Awards', navService: 'Service',
      pageTitle: 'Academic homepage', paper: 'Paper', code: 'Code',
      navigation: 'Main navigation', language: 'Language', menu: 'Toggle navigation',
      skip: 'Skip to content', swapPortrait: 'Switch portrait',
      fantasyMode: 'Enter fantasy mode', workMode: 'Back to work mode',
      viewFigure: 'View full figure:', figurePreview: 'Figure preview', closePreview: 'Close preview',
    },
    zh: {
      about: '个人介绍', publications: '学术论文', honors: '荣誉奖项',
      service: '学术服务', navHonors: '荣誉奖项', navService: '学术服务',
      pageTitle: '个人学术主页', paper: '论文', code: '代码',
      navigation: '主导航', language: '语言切换', menu: '展开或收起导航',
      skip: '跳转到正文', swapPortrait: '切换头像',
      fantasyMode: '进入幻想模式', workMode: '回到工作模式',
      viewFigure: '查看完整框架图：', figurePreview: '论文插图预览', closePreview: '关闭预览',
    },
  };

  // Only biography.researchHtml deliberately accepts markup; other content is escaped.
  function escapeHtml(value) {
    const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return String(value).replace(/[&<>"']/g, character => entities[character]);
  }

  function externalLink(text, href) {
    return `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(text)}</a>`;
  }

  function icon(path) {
    return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
  }

  function renderLanguages() {
    return `<div class="language-switch" role="group" aria-label="${labels[lang].language}">
      <button type="button" data-lang="en" lang="en" aria-pressed="${lang === 'en'}">EN</button>
      <button type="button" data-lang="zh" lang="zh-CN" aria-pressed="${lang === 'zh'}">中文</button>
    </div>`;
  }

  function renderNavigation(className) {
    const text = labels[lang];
    const items = [text.about, text.publications, text.navHonors, text.navService];
    const id = className === 'header-nav' ? 'id="primary-navigation"' : '';
    return `<nav class="${className}" ${id} aria-label="${text.navigation}">
      ${sections.map((section, index) => `<a href="#${section}" ${index === 0 ? 'class="is-active" aria-current="location"' : ''}>${items[index]}</a>`).join('')}
    </nav>`;
  }

  function renderHeader() {
    return `<header class="site-header ${sidebarLayout ? 'mobile-b-header' : ''}">
      <div class="header-inner">
        <a class="wordmark" href="#about">${escapeHtml(profile.name)} (${escapeHtml(profile.nameZh)})</a>
        ${renderNavigation('header-nav')}
        ${renderLanguages()}
        <button class="menu-toggle" type="button" aria-expanded="false" aria-controls="primary-navigation" aria-label="${labels[lang].menu}">${icon(menuIcon)}</button>
      </div>
    </header>`;
  }

  function portraitMode() {
    return labels[lang][alternatePortrait ? 'workMode' : 'fantasyMode'];
  }

  function renderPortrait() {
    const image = alternatePortrait ? profile.avatarAlt : profile.avatar;
    return `<button class="portrait" type="button" tabindex="-1" aria-label="${labels[lang].swapPortrait}: ${portraitMode()}" aria-pressed="${alternatePortrait}">
      <img class="avatar" src="assets/${escapeHtml(image)}" alt="${escapeHtml(lang === 'zh' ? profile.nameZh : profile.name)}" width="444" height="592" draggable="false">
    </button>`;
  }

  function renderIntroduction() {
    const text = biography[lang];
    const advisor = externalLink(profile.advisorLabel[lang], profile.advisorUrl);
    const separator = lang === 'en' ? ' ' : '';
    const punctuation = lang === 'en' ? '. ' : '。';
    return `<section class="about-section" id="about">
      <div class="intro">
        <h1>${escapeHtml(profile.name)} <span class="alternate-name">(${escapeHtml(profile.nameZh)})</span></h1>
        <p>${escapeHtml(text.introduction)}${separator}${advisor}${punctuation}${escapeHtml(text.education)}</p>
        <p>${text.researchHtml}</p>
        <p>${escapeHtml(text.invitation)}</p>
        <div class="profile-links">
          <a href="mailto:${escapeHtml(profile.email)}">Email</a>
          ${externalLink('Scholar', profile.scholar)}
          ${externalLink('GitHub', profile.github)}
        </div>
      </div>
      ${renderPortrait()}
    </section>`;
  }

  function renderHeading(id, text) {
    return `<div class="section-heading"><h2 id="${id}">${text}</h2></div>`;
  }

  function renderAuthors(paper) {
    const names = paper.authors.map(name => {
      const star = paper.equalContribution?.includes(name) ? '<sup>*</sup>' : '';
      const author = `${escapeHtml(name)}${star}`;
      return name === profile.name ? `<strong>${author}</strong>` : `<span class="co-author">${author}</span>`;
    }).join(', ');
    const note = paper.equalContribution?.length ? ' <span class="author-note">(* Equal contribution.)</span>' : '';
    return names + note;
  }

  function renderPaper(paper) {
    const rank = paper.ccfRank ? ` <span aria-hidden="true">|</span> CCF-${escapeHtml(paper.ccfRank)}` : '';
    return `<article class="publication" data-paper="${escapeHtml(paper.id)}">
      <button class="figure-link" type="button" aria-haspopup="dialog" aria-label="${labels[lang].viewFigure} ${escapeHtml(paper.title)}">
        <img src="assets/${escapeHtml(paper.image)}" alt="${escapeHtml(paper.title)} — framework" width="300" height="160">
      </button>
      <div class="paper-copy">
        <h3>${externalLink(paper.title, paper.paper)}</h3>
        <p class="authors">${renderAuthors(paper)}</p>
        <div class="venue">
          <span class="venue-label">${escapeHtml(paper.venue)}${rank}</span>
          <span class="venue-name">${escapeHtml(paper.venueFull)}</span>
        </div>
        <p class="summary">${escapeHtml(paper[lang])}</p>
        <div class="paper-links">
          ${externalLink(labels[lang].paper, paper.paper)}
          ${paper.code ? `<span>/</span>${externalLink(labels[lang].code, paper.code)}` : ''}
        </div>
      </div>
    </article>`;
  }

  function renderSections() {
    const text = labels[lang];
    const chronological = [...papers].sort((a, b) => b.year - a.year);
    return `<section id="publications" aria-labelledby="publications-title">
      ${renderHeading('publications-title', text.publications)}
      <div class="publication-list">${chronological.map(renderPaper).join('')}</div>
    </section>
    <div class="lower-sections">
      <section id="honors" aria-labelledby="honors-title">
        ${renderHeading('honors-title', text.honors)}
        <ul class="honor-list">${honors.map(honor => `<li><time>${escapeHtml(honor.year)}</time><span>${escapeHtml(honor[lang])}</span></li>`).join('')}</ul>
      </section>
      <section id="service" aria-labelledby="service-title">
        ${renderHeading('service-title', text.service)}
        <ul class="service-list">${service[lang].map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </section>
    </div>
    <footer class="updated-note">Updated <time datetime="${escapeHtml(profile.updated)}">${escapeHtml(profile.updated.replace('-', '.'))}</time></footer>`;
  }

  function setMenu(button, open) {
    button.closest('.header-inner').querySelector('.header-nav').classList.toggle('is-open', open);
    button.setAttribute('aria-expanded', String(open));
    button.querySelector('path').setAttribute('d', open ? closeIcon : menuIcon);
  }

  function closeMenu(returnFocus = false) {
    document.querySelectorAll('.menu-toggle[aria-expanded="true"]').forEach(button => {
      setMenu(button, false);
      if (returnFocus) button.focus();
    });
  }

  function openFigure(button) {
    if (figurePreview?.open) return;
    closeMenu();
    const source = button.querySelector('img');
    const dialog = document.createElement('dialog');
    dialog.className = 'figure-preview';
    dialog.setAttribute('aria-label', labels[lang].figurePreview);
    dialog.innerHTML = `<button class="preview-close" type="button" aria-label="${labels[lang].closePreview}">${icon(closeIcon)}</button><img>`;
    const image = dialog.querySelector('img');
    image.src = source.src;
    image.alt = source.alt;
    dialog.querySelector('.preview-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', event => {
      if (event.target === dialog) dialog.close();
    });
    dialog.addEventListener('close', () => {
      dialog.remove();
      if (figurePreview === dialog) {
        figurePreview = null;
        document.body.classList.remove('preview-open');
      }
    });
    document.body.append(dialog);
    figurePreview = dialog;
    document.body.classList.add('preview-open');
    dialog.showModal();
  }

  function togglePortrait(button) {
    alternatePortrait = !alternatePortrait;
    button.setAttribute('aria-pressed', String(alternatePortrait));
    button.setAttribute('aria-label', `${labels[lang].swapPortrait}: ${portraitMode()}`);
    button.querySelector('img').src = `assets/${alternatePortrait ? profile.avatarAlt : profile.avatar}`;
    button.blur();
  }

  function changeLanguage(nextLanguage) {
    if (nextLanguage === lang) return;
    const top = scrollY;
    lang = nextLanguage;
    const url = new URL(location);
    url.searchParams.set('lang', lang);
    history.replaceState(null, '', url);
    render();
    scrollTo({ top, behavior: 'instant' });
  }

  function bindEvents() {
    document.querySelectorAll('.figure-link').forEach(button => {
      button.addEventListener('click', () => openFigure(button));
    });
    const portrait = document.querySelector('.portrait');
    portrait.addEventListener('mousedown', event => event.preventDefault());
    portrait.addEventListener('focus', () => portrait.blur());
    portrait.addEventListener('click', () => togglePortrait(portrait));
    document.querySelectorAll('[data-lang]').forEach(button => {
      button.addEventListener('click', () => changeLanguage(button.dataset.lang));
    });
    document.querySelectorAll('.menu-toggle').forEach(button => {
      button.addEventListener('click', () => setMenu(button, button.getAttribute('aria-expanded') !== 'true'));
    });
    document.querySelectorAll('nav a').forEach(link => {
      link.addEventListener('click', () => closeMenu());
    });

    navigationObserver?.disconnect();
    navigationObserver = new IntersectionObserver(entries => {
      const section = entries.find(entry => entry.isIntersecting)?.target;
      if (!section) return;
      document.querySelectorAll('nav a').forEach(link => {
        const active = link.getAttribute('href') === '#' + section.id;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, { rootMargin: '-10% 0px -65% 0px', threshold: 0 });
    sections.forEach(id => navigationObserver.observe(document.getElementById(id)));
  }

  function render() {
    figurePreview?.close();
    document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
    document.title = `${profile.name} · ${labels[lang].pageTitle}`;
    const content = renderIntroduction() + renderSections();
    const page = sidebarLayout
      ? `<div class="side-layout"><aside class="side-rail">${renderNavigation('side-nav')}${renderLanguages()}</aside><main class="side-main" id="main">${content}</main></div>`
      : `<main class="page" id="main">${content}</main>`;
    app.innerHTML = `<a class="skip-link" href="#main">${labels[lang].skip}</a>${renderHeader()}${page}`;
    bindEvents();
  }

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !figurePreview?.open) closeMenu(true);
  });
  document.addEventListener('click', event => {
    if (!event.target.closest('.site-header')) closeMenu();
  });
  compactViewport.addEventListener('change', () => closeMenu());
  render();
})();
