import { describe, it, expect } from 'vitest';
import Settings from './Settings.vue';
import { mountApp } from '../test/test-helpers';

describe('Settings.vue', () => {
  it('should mount without errors', () => {
    const wrapper = mountApp(Settings);

    expect(wrapper.exists()).toBe(true);
  });

  it('should render settings panel', () => {
    const wrapper = mountApp(Settings);

    expect(wrapper.find('.settings-panel').exists()).toBe(true);
    expect(wrapper.find('h2').text()).toBe('System Settings');
  });

  it('should render Common section with Modem Mode', () => {
    const wrapper = mountApp(Settings);

    expect(wrapper.find('.section-title').exists()).toBe(true);
    // 0: Language, 1: Modem Mode (Common), 2: Sender
    expect(wrapper.findAll('.section-title')[1].text()).toBe('Modem Mode (PHY)');

    // Modem Mode セレクタが存在する
    expect(wrapper.find('.mode-selector').exists()).toBe(true);
  });

  it('should render two Modem Mode buttons', () => {
    const wrapper = mountApp(Settings);

    const modeButtons = wrapper.findAll('.mode-btn');
    expect(modeButtons.length).toBe(2);
    expect(modeButtons[0].text()).toContain('DSSS (Lightweight, Low-CPU, Low-speed)');
    expect(modeButtons[1].text()).toContain('M-ary (High-performance, High-load, High-speed)');
  });

  it('should render Debug Mode checkbox', () => {
    const wrapper = mountApp(Settings);

    // checkboxを見つける
    const checkboxes = wrapper.findAll('input[type="checkbox"]');
    expect(checkboxes.length).toBeGreaterThan(0);
  });

  it('should render Sender section with Randomized Seq', () => {
    const wrapper = mountApp(Settings);

    const sectionTitles = wrapper.findAll('.section-title');
    expect(sectionTitles.length).toBeGreaterThan(2);
    expect(sectionTitles[2].text()).toBe('Sender');
  });

  it('should render Randomized Seq checkbox in Sender section', () => {
    const wrapper = mountApp(Settings);

    // 2つのcheckboxがあるべき（Debug ModeとRandomized Seq）
    const checkboxes = wrapper.findAll('input[type="checkbox"]');
    expect(checkboxes.length).toBe(2);
  });

  it('should have Mary mode active by default', () => {
    const wrapper = mountApp(Settings);

    const modeButtons = wrapper.findAll('.mode-btn');
    expect(modeButtons[1].classes()).toContain('active');
  });

  it('should enable Modem Mode buttons when runtime is not busy', () => {
    const wrapper = mountApp(Settings);

    const modeButtons = wrapper.findAll('.mode-btn');
    modeButtons.forEach(btn => {
      expect(btn.attributes('disabled')).toBeUndefined();
    });
  });
});
